"""LSTM model for processing time series data."""

from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.models import Model
from typing import Tuple
from ...config import LSTM_UNITS, DENSE_UNITS

def build_lstm_model(input_shape: Tuple[int, int]) -> Tuple[Input, Dense]:
    """
    Build LSTM model for processing time series data.
    
    Args:
        input_shape: Shape of input sequences (timesteps, features)
        
    Returns:
        Tuple containing:
        - Input layer
        - Output tensor
    """
    input_layer = Input(shape=input_shape)
    
    # First LSTM layer with return sequences
    x = LSTM(LSTM_UNITS[0], return_sequences=True)(input_layer)
    
    # Second LSTM layer
    x = LSTM(LSTM_UNITS[1], return_sequences=False)(x)
    
    # Dense layer
    x = Dense(DENSE_UNITS[1], activation='relu')(x)
    
    return input_layer, x