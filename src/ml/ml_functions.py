import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR, StepLR, CosineAnnealingLR
from sklearn.metrics import mean_absolute_error, mean_squared_error


class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2):
        """
        Modelo LSTM para prever séries temporais com múltiplas features.

        Parâmetros:
            input_size (int): Número de features de entrada.
            hidden_layer_size (int): Tamanho da camada oculta da LSTM.
            output_size (int): Tamanho da saída (normalmente 1 para prever apenas o preço).
            num_layers (int): Número de camadas LSTM empilhadas.
            dropout (float): Taxa de dropout para regularização.
        """
        super(StockLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # LSTM empilhada com suporte para múltiplas features
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout,
                            batch_first=True)

        # Dropout para regularização
        self.dropout = nn.Dropout(dropout)

        # Camadas densas adicionais
        self.linear_hidden = nn.Linear(hidden_layer_size, hidden_layer_size // 2)
        self.relu_hidden = nn.ReLU()
        self.linear_post = nn.Linear(hidden_layer_size // 2, output_size)

    def forward(self, input_seq):
        """
        Forward pass do modelo.

        Parâmetros:
            input_seq (tensor): Tensor de entrada com shape (batch_size, seq_length, input_size).

        Retorna:
            predictions (tensor): Tensor de saída com shape (batch_size, output_size).
        """
        # Estados iniciais da LSTM
        h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)
        c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_layer_size).to(input_seq.device)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(input_seq, (h0, c0))

        # Usar o estado oculto final (última camada)
        combined_hidden = h_n[-1]

        # Aplicar dropout
        combined_hidden = self.dropout(combined_hidden)

        # Passar pelas camadas densas adicionais
        hidden_output = self.relu_hidden(self.linear_hidden(combined_hidden))
        predictions = self.linear_post(hidden_output)

        return predictions

def calculate_metrics(y_pred, y_true):
    """
    Calcula métricas relevantes para problemas de regressão:
    MAE, MSE e RMSE.

    Parâmetros:
        y_pred: Tensores com as predições do modelo.
        y_true: Tensores com os valores reais.

    Retorna:
        Um dicionário com as métricas calculadas.
    """
    mae = torch.mean(torch.abs(y_pred - y_true)).item()
    mse = torch.mean((y_pred - y_true) ** 2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    return {"MAE": mae, "MSE": mse, "RMSE": rmse}


def train_epoch(model, dataloader, criterion, optimizer=None, is_training=True):
    """
    Treina ou avalia o modelo para uma época.

    Parâmetros:
        model: Modelo a ser treinado ou avaliado.
        dataloader: DataLoader para o conjunto de dados (treinamento ou validação).
        criterion: Função de perda.
        optimizer: Otimizador (apenas para treinamento).
        is_training: Define se é uma etapa de treinamento ou avaliação.

    Retorna:
        A perda média e as métricas de avaliação (MAE, MSE, RMSE).
    """
    epoch_loss = 0
    all_y_pred = []
    all_y_true = []

    if is_training:
        model.train()
    else:
        model.eval()

    for x_batch, y_batch in dataloader:
        if is_training:
            optimizer.zero_grad()

        # Garantir que os tensores estejam no dispositivo correto
        x_batch = x_batch.to(next(model.parameters()).device)
        y_batch = y_batch.to(next(model.parameters()).device)

        # Forward pass
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)

        if is_training:
            # Backward pass e otimização
            loss.backward()
            optimizer.step()

        # Acumular métricas e perdas
        epoch_loss += loss.item()
        all_y_pred.append(y_pred.detach())
        all_y_true.append(y_batch.detach())

    # Calcular a perda média
    epoch_loss /= len(dataloader)

    # Combinar predições e valores reais para métricas
    all_y_pred = torch.cat(all_y_pred)
    all_y_true = torch.cat(all_y_true)

    metrics = calculate_metrics(all_y_pred, all_y_true)
    return epoch_loss, metrics


def train_model(
    model, train_loader, val_loader, epochs=150, lr=0.001, weight_decay=0.01,
    scheduler_type="step", early_stopping_patience=10,
    patience=5, factor=0.5, cyclic_base_lr=1e-5, cyclic_max_lr=0.0005,
    step_size_up=10, step_size=30, gamma=0.5, t_max=50
):
    """
    Treina o modelo para um problema de regressão.

    Parâmetros:
        model: O modelo a ser treinado.
        train_loader: DataLoader para os dados de treino.
        val_loader: DataLoader para os dados de validação.
        epochs: Número de épocas de treinamento.
        lr: Taxa de aprendizado inicial.
        weight_decay: Regularização L2.
        scheduler_type: Tipo de scheduler ('reduce_on_plateau', 'cyclic', 'step', 'cosine_annealing').
        early_stopping_patience: Paciência para early stopping.
        Parâmetros específicos para os schedulers:
            patience, factor, cyclic_base_lr, cyclic_max_lr, step_size_up, step_size, gamma, t_max.

    Retorna:
        O modelo treinado.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Configuração do scheduler
    if scheduler_type == "reduce_on_plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=patience, factor=factor)
    elif scheduler_type == "cyclic":
        scheduler = CyclicLR(optimizer, base_lr=cyclic_base_lr, max_lr=cyclic_max_lr, step_size_up=step_size_up,
                             mode='triangular2')
    elif scheduler_type == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == "cosine_annealing":
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
    else:
        raise ValueError(
            "Tipo de scheduler inválido. Escolha entre 'reduce_on_plateau', 'cyclic', 'step', ou 'cosine_annealing'.")

    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # Treinamento
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, is_training=True)

        # Validação
        val_loss, val_metrics = train_epoch(model, val_loader, criterion, is_training=False)

        # Atualizar o scheduler
        if scheduler_type == "reduce_on_plateau":
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # Exibição dos resultados
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"Train RMSE: {train_metrics['RMSE']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f} | "
                  f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_model_state = model.state_dict()
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # Restaurar o melhor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def evaluate_model(model, X_train, y_train, X_test, y_test, scaler):
    """
    Avalia o modelo usando os conjuntos de treino e teste.

    Parâmetros:
        model: Modelo treinado.
        X_train: Tensor de entrada de treino.
        y_train: Tensor de saída real de treino.
        X_test: Tensor de entrada de teste.
        y_test: Tensor de saída real de teste.
        scaler: Objeto MinMaxScaler usado para normalizar os dados.

    Retorna:
        train_preds: Previsões para o conjunto de treino (desnormalizadas).
        test_preds: Previsões para o conjunto de teste (desnormalizadas).
        actual: Valores reais desnormalizados de treino e teste.
    """
    model.eval()  # Modo de avaliação
    with torch.no_grad():
        # Garantir que os tensores estejam no dispositivo correto
        X_train = X_train.to(next(model.parameters()).device)
        y_train = y_train.to(next(model.parameters()).device)
        X_test = X_test.to(next(model.parameters()).device)
        y_test = y_test.to(next(model.parameters()).device)

        # Previsões
        train_preds = model(X_train).cpu().numpy()
        test_preds = model(X_test).cpu().numpy()

    # Ajustar as previsões para o formato esperado pelo scaler
    num_features = scaler.min_.shape[0]  # Número total de features usadas no scaler
    train_preds_full = np.zeros((train_preds.shape[0], num_features))
    test_preds_full = np.zeros((test_preds.shape[0], num_features))

    train_preds_full[:, 0] = train_preds.flatten()  # Inserir previsões na coluna correspondente ao 'Close'
    test_preds_full[:, 0] = test_preds.flatten()

    # Inverter a normalização apenas para a coluna 'Close'
    train_preds = scaler.inverse_transform(train_preds_full)[:, 0]
    test_preds = scaler.inverse_transform(test_preds_full)[:, 0]

    # Inverter a normalização para os valores reais
    y_train_full = np.zeros((y_train.shape[0], num_features))
    y_test_full = np.zeros((y_test.shape[0], num_features))
    y_train_full[:, 0] = y_train.cpu().numpy().flatten()
    y_test_full[:, 0] = y_test.cpu().numpy().flatten()

    actual_train = scaler.inverse_transform(y_train_full)[:, 0]
    actual_test = scaler.inverse_transform(y_test_full)[:, 0]

    # Métricas
    #train_mae = mean_absolute_error(actual_train, train_preds)
    #test_mae = mean_absolute_error(actual_test, test_preds)
    #train_rmse = np.sqrt(mean_squared_error(actual_train, train_preds))
    #test_rmse = np.sqrt(mean_squared_error(actual_test, test_preds))
    #train_mape = np.mean(np.abs((actual_train - train_preds) / actual_train)) * 100
    #test_mape = np.mean(np.abs((actual_test - test_preds) / actual_test)) * 100

    # Retornar previsões e valores reais
    return torch.tensor(train_preds), torch.tensor(test_preds), torch.tensor(np.concatenate([actual_train, actual_test]))

def future_predictions(model, last_sequence, future_days, scaler):
    """
    Generate future predictions using the trained model.

    Parameters:
        model (torch.nn.Module): The trained LSTM model.
        last_sequence (torch.Tensor): The last sequence of data points to use as input for predictions.
        future_days (int): The number of days to predict into the future.
        scaler (scaler object): The scaler used for inverse transformation.

    Returns:
        np.array: Future predictions, inverse transformed to the original scale.
    """
    predictions = []
    model.eval()

    # Garantir que last_sequence tenha exatamente 3 dimensões: (1, sequence_length, num_features)
    if last_sequence.dim() == 2:  # Caso last_sequence esteja com 2D (seq_length, num_features)
        last_sequence = last_sequence.unsqueeze(0)  # Adicionar dimensão de batch

    with torch.no_grad():
        for _ in range(future_days):
            # Fazer a previsão
            future_pred = model(last_sequence)  # Saída esperada: (batch_size, output_size)

            # Extraindo a previsão
            next_pred = future_pred[0, 0]  # Saída é escalar

            # Armazenar a previsão
            predictions.append(next_pred.item())

            # Ajustar `next_pred` para o formato correto: (1, 1, num_features)
            next_pred_tensor = next_pred.unsqueeze(0).unsqueeze(0).repeat(1, 1, last_sequence.size(-1))

            # Atualizar `last_sequence` para incluir a nova previsão
            last_sequence = torch.cat((last_sequence[:, 1:, :], next_pred_tensor), dim=1)

    # Criar um array de shape (future_days, num_features) para o inverso
    num_features = scaler.scale_.shape[0]
    predictions_array = np.zeros((future_days, num_features))
    predictions_array[:, 0] = predictions  # Preencher apenas a coluna correspondente ao `Close`

    # Transformar as previsões de volta para a escala original
    inverse_transformed = scaler.inverse_transform(predictions_array)
    return inverse_transformed[:, 0]  # Retornar apenas a coluna `Close`
