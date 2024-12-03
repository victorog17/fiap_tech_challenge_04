import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler


def fetch_yfinance_data(ticker, period):
    """
    Fetches financial asset data using yfinance, with error handling.

    Parameters:
        ticker (str): Financial asset symbol.
        period (str): Data collection period (e.g., '1mo', '1y', '10y').

    Returns:
        pd.DataFrame: DataFrame with historical data containing the 'Close' column.
    """
    try:
        data = yf.download(tickers=ticker, period=period)
        if data.empty:
            raise ValueError(f"No data was downloaded for ticker '{ticker}' in the period '{period}'.")
        return data
    except Exception as e:
        raise ValueError(f"Error fetching data: {e}")


def add_technical_indicators(df, feature_columns=None, sma_period=20, rsi_period=14,
                             macd_span_short=12, macd_span_long=26):
    """
    Adds technical indicators to the DataFrame with dynamic parameter calculation
    and preserves only the desired columns.

    Parameters:
        df (pd.DataFrame): DataFrame with historical data (must contain the 'Close' column).
        feature_columns (list): List of columns to be retained from the original DataFrame.
        sma_period (int): Period for calculating the Simple Moving Average (SMA).
        rsi_period (int): Period for calculating the Relative Strength Index (RSI).
        macd_span_short (int): Short period for calculating the Exponential Moving Average (EMA) of the MACD.
        macd_span_long (int): Long period for calculating the EMA of the MACD.

    Returns:
        pd.DataFrame: DataFrame with the added technical indicators.
    """
    if 'Close' not in df.columns:
        raise KeyError("The 'Close' column was not found in the DataFrame.")

    df['Close'] = df['Close'].astype(float)

    # Ensure 'Close' is one-dimensional
    df['Close'] = pd.Series(df['Close'].values.flatten(), index=df.index)

    # SMA (Simple Moving Average)
    df[f'sma_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()

    # RSI (Relative Strength Index)
    delta = df['Close'].diff(1)
    gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
    loss = -delta.clip(upper=0).rolling(window=rsi_period).mean()
    rs = gain / loss
    df[f'rsi_{rsi_period}'] = 100 - (100 / (1 + rs))

    # MACD (Moving Average Convergence Divergence)
    ema_short = df['Close'].ewm(span=macd_span_short, adjust=False).mean()
    ema_long = df['Close'].ewm(span=macd_span_long, adjust=False).mean()
    df['macd'] = ema_short - ema_long

    df = df.dropna()

    # Retain only the specified columns and the new indicators
    if feature_columns is not None:
        columns_to_keep = feature_columns + [f'sma_{sma_period}', f'rsi_{rsi_period}', 'macd']
        df = df[columns_to_keep]

    new_features = df.columns

    return df, new_features

def preprocess_data(data, feature_columns):
    """
    Normalize data with multiple features using MinMaxScaler and ensure data consistency.

    Parameters:
        data (pd.DataFrame): Input data to normalize. Should be a DataFrame with multiple features.
        feature_columns (list): List of columns to normalize.

    Returns:
        scaled_data (np.array): Normalized data (NumPy array with shape [n_samples, n_features]).
        scaler (MinMaxScaler): Fitted scaler for inverse transformations.
    """
    # Select only the relevant columns (features)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("The 'data' parameter must be a pandas DataFrame.")
    if not all(col in data.columns for col in feature_columns):
        raise ValueError("Some columns specified in 'feature_columns' are not present in the DataFrame.")

    data = data[feature_columns].copy()

    # Ensure the data does not contain NaN or infinite values
    if data.isnull().any().any():
        raise ValueError(f"The data contains NaN values. Problematic indices:\n{data.isnull().sum()}")
    if np.isinf(data.values).any():
        raise ValueError("The data contains infinite values.")

    # Initialize and fit the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    return scaled_data, scaler

