import torch
import os
import time
import psutil
import numpy as np


def get_device(framework="pytorch"):
    """
    Returns the appropriate device for PyTorch or Keras/TensorFlow.

    Parameters:
        framework (str): The framework being used, either "pytorch" or "keras".

    Returns:
        device (torch.device or str): The device to be used, either "cuda" or "cpu" for PyTorch,
                                      or "/GPU:0" or "/CPU:0" for Keras/TensorFlow.
    """
    if framework == "pytorch":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for PyTorch: {device}")
        return device
    else:
        raise ValueError("Unsupported framework. Choose 'pytorch' or 'keras'.")
    
def save_model_local(model, path='model.pth'):
    """
    Salva o modelo treinado localmente no formato PyTorch.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"Modelo salvo em {path}")


def save_scaler_torch(scaler, path):
    """
    Salva o scaler usando torch.save.

    Parameters:
        scaler: O objeto scaler treinado (e.g., MinMaxScaler).
        path (str): Caminho onde o scaler será salvo.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(scaler, path)
    print(f"Scaler salvo em {path}")

def create_sequences(data, seq_length=60, target_column=0):
    """
    Split data into sequences and targets based on a specified sequence length.

    Parameters:
        data (array-like): Data to split into sequences. Assumes multiple features (columns).
        seq_length (int): Length of each sequence.
        target_column (int): Index of the column to use as the target.

    Returns:
        sequences (np.array): Data sequences of shape (num_sequences, seq_length, num_features).
        targets (np.array): Target values of shape (num_sequences,).
    """
    # Ensure the data does not contain NaN or infinite values
    assert not np.any(np.isnan(data)), "The data contains NaN values."
    assert not np.any(np.isinf(data)), "The data contains infinite values."

    # Check if the data has sufficient length
    if len(data) <= seq_length:
        raise ValueError("The data must be longer than the sequence length (seq_length).")

    # Ensure the target column exists
    if target_column >= data.shape[1]:
        raise ValueError(f"The target column index ({target_column}) is greater than the number of columns ({data.shape[1]}).")

    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        # Extract sequence of all features
        seq = data[i:i + seq_length]
        # Extract the target value only from the specified column
        target = data[i + seq_length, target_column]

        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)

def prepare_tensors_pytorch(data_sequences, data_targets, device):
    """
    Convert data sequences and targets into PyTorch tensors with the correct dimensions.

    Parameters:
        data_sequences (np.array): Input data sequences.
        data_targets (np.array): Target values corresponding to the sequences.
        device (torch.device): Device to allocate tensors (CPU or GPU).

    Returns:
        X (torch.Tensor): Prepared input tensors with 3 dimensions (batch_size, sequence_length, input_size).
        y (torch.Tensor): Prepared target tensors with 2 dimensions (batch_size, target_size).
    """
    # Ensure that X has 3 dimensions: (batch_size, sequence_length, input_size)
    X = torch.tensor(data_sequences, dtype=torch.float32).to(device)

    # Ensure that y has 2 dimensions: (batch_size, target_size)
    y = torch.tensor(data_targets, dtype=torch.float32).view(-1, 1).to(device)

    return X, y

def monitor_performance(model, X_test, y_test, scaler):
    """
    Monitora o desempenho do modelo em termos de tempo de resposta e utilização de recursos.
    """
    model.eval()
    inference_times = []
    cpu_usages = []
    memory_usages = []

    with torch.no_grad():
        for i in range(len(X_test)):
            start_time = time.time()

            # Inferência do modelo
            _ = model(X_test[i:i+1])

            # Medir o tempo de inferência
            inference_time = time.time() - start_time
            inference_times.append(inference_time)

            # Capturar métricas de recursos
            cpu_usage = psutil.cpu_percent(interval=None)
            memory_usage = psutil.virtual_memory().percent
            cpu_usages.append(cpu_usage)
            memory_usages.append(memory_usage)

    # Calcular métricas médias
    avg_inference_time = sum(inference_times) / len(inference_times)
    avg_cpu_usage = sum(cpu_usages) / len(cpu_usages)
    avg_memory_usage = sum(memory_usages) / len(memory_usages)

    print(f"Average Inference Time: {avg_inference_time:.6f} seconds")
    print(f"Average CPU Usage: {avg_cpu_usage:.2f}%")
    print(f"Average Memory Usage: {avg_memory_usage:.2f}%")