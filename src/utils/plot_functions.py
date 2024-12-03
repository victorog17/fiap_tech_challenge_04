import numpy as np
import matplotlib.pyplot as plt

def plot_all(actual, train_preds, test_preds, future_preds, seq_length, future_days):
    """
    Plota os valores reais, as previsões de treino, teste e as previsões futuras.

    Parâmetros:
    - actual (np.array): Série de preços reais.
    - train_preds (np.array): Previsões do conjunto de treino.
    - test_preds (np.array): Previsões do conjunto de teste.
    - future_preds (np.array): Previsões para os dias futuros.
    - seq_length (int): Comprimento da sequência usada no modelo.
    - future_days (int): Número de dias para previsões futuras.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(np.arange(seq_length, len(actual)), actual[seq_length:], label="Actual Prices", color="blue")
    plt.plot(np.arange(seq_length, seq_length + len(train_preds)), train_preds, label="Training Predictions", color="orange")
    test_start_idx = seq_length + len(train_preds)
    plt.plot(np.arange(test_start_idx, test_start_idx + len(test_preds)), test_preds, label="Testing Predictions", color="green")
    future_start_idx = test_start_idx + len(test_preds)
    plt.plot(np.arange(future_start_idx, future_start_idx + future_days), future_preds, label="Future Predictions", color="red", linestyle="--")
    plt.title("Stock Price Predictions")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_residuals(actual, predictions):
    residuals = actual - predictions
    plt.figure(figsize=(12, 6))
    plt.plot(residuals, label="Residuals")
    plt.title("Residuals Over Time")
    plt.xlabel("Time")
    plt.ylabel("Error")
    plt.legend()
    plt.show()

def plot_residual_distribution(residuals):
    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30, edgecolor='black')
    plt.title("Distribution of Residuals")
    plt.xlabel("Residual Value")
    plt.ylabel("Frequency")
    plt.show()

def plot_train_test_predictions(actual, train_preds, test_preds):
    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Prices", color="blue")
    plt.plot(train_preds, label="Training Predictions", color="orange")
    plt.plot(range(len(train_preds), len(train_preds) + len(test_preds)), test_preds, label="Testing Predictions", color="green")
    plt.title("Train vs. Test Predictions")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

def plot_confidence_interval(actual, predictions, residuals):
    confidence_interval = 1.96 * residuals.std()  # 95% Confidence Interval

    # Garantir que predictions seja um array NumPy unidimensional
    predictions = np.array(predictions).flatten()

    plt.figure(figsize=(12, 6))
    plt.plot(actual, label="Actual Prices", color="blue")
    plt.plot(predictions, label="Predictions", color="red")
    plt.fill_between(
        range(len(predictions)),
        predictions - confidence_interval,
        predictions + confidence_interval,
        color='red',
        alpha=0.2,
        label="Confidence Interval (95%)"
    )
    plt.title("Predictions with Confidence Interval")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()


def plot_autocorrelation(residuals, max_lag=40):
    """
    Plota a autocorrelação dos resíduos para os primeiros lags até max_lag.

    Parâmetros:
    - residuals (np.array): Array de resíduos do modelo.
    - max_lag (int): Número máximo de lags para calcular a autocorrelação.
    """
    # Calcula a média dos resíduos
    mean_residuals = np.mean(residuals)
    # Inicializa uma lista para armazenar as autocorrelações
    autocorrelations = []

    for lag in range(1, max_lag + 1):
        # Calcula o numerador da fórmula de autocorrelação
        numerator = np.sum((residuals[:-lag] - mean_residuals) * (residuals[lag:] - mean_residuals))
        # Calcula o denominador (variância)
        denominator = np.sum((residuals - mean_residuals) ** 2)

        # Verifica se o denominador é diferente de zero
        if denominator != 0:
            autocorrelation = numerator / denominator
        else:
            autocorrelation = 0

        autocorrelations.append(autocorrelation)

    # Plot do gráfico de autocorrelação
    plt.figure(figsize=(10, 5))
    plt.bar(range(1, max_lag + 1), autocorrelations, color='blue')
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.title("Residual Autocorrelation")
    plt.show()

def plot_historical_and_future(actual, future_preds):
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(actual)), actual, label="Historical Data", color="blue")
    plt.plot(range(len(actual), len(actual) + len(future_preds)), future_preds, label="Future Predictions", color="red")
    plt.title("Historical Data and Future Predictions")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.show()
