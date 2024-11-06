"""Utility functions for data preprocessing."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple
import pandas as pd
from ...config import WINDOW_SIZE

def prepare_time_series_data(
    data: pd.DataFrame,
    window_size: int = WINDOW_SIZE
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Prepare time series data for LSTM input.
    
    Args:
        data: DataFrame containing stock data
        window_size: Size of the sliding window
        
    Returns:
        Tuple containing:
        - X: Input sequences
        - y: Target values
        - scaler: Fitted MinMaxScaler object
    """
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    
    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
        y.append(scaled_data[i])
        
    return np.array(X), np.array(y), scaler

def align_data_lengths(
    X_cnn: np.ndarray,
    X_lstm: np.ndarray,
    sentiment_data: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensure all input data arrays have matching lengths.
    
    Args:
        X_cnn: CNN input data
        X_lstm: LSTM input data
        sentiment_data: Sentiment scores
        y: Target values
        
    Returns:
        Tuple of aligned arrays
    """
    min_length = min(len(y), len(X_cnn), len(X_lstm), len(sentiment_data))
    
    return (X_cnn[:min_length],
            X_lstm[:min_length],
            sentiment_data[:min_length],
            y[:min_length])