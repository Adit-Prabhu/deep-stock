"""Hybrid model combining CNN, LSTM, and sentiment analysis."""

from tensorflow.keras.layers import Input, Dense, concatenate
from tensorflow.keras.models import Model
from typing import Tuple
import numpy as np
from .cnn import build_cnn_model
from .lstm import build_lstm_model
from config import (CHART_SIZE, DENSE_UNITS, BATCH_SIZE,
                      EPOCHS, VALIDATION_SPLIT)

def build_combined_model(
    cnn_input_shape: Tuple[int, int, int],
    lstm_input_shape: Tuple[int, int]
) -> Model:
    """
    Build hybrid model combining CNN, LSTM, and sentiment analysis.
    
    Args:
        cnn_input_shape: Shape of CNN input
        lstm_input_shape: Shape of LSTM input
        
    Returns:
        Combined Keras model
    """
    # Build individual components
    cnn_input, cnn_model = build_cnn_model(cnn_input_shape)
    lstm_input, lstm_model = build_lstm_model(lstm_input_shape)
    
    # Sentiment input
    sentiment_input = Input(shape=(1,), name='sentiment_input')
    sentiment_dense = Dense(DENSE_UNITS[2], activation='relu')(sentiment_input)
    
    # Combine all inputs
    combined = concatenate([cnn_model, lstm_model, sentiment_dense])
    output = Dense(1, activation='linear')(combined)
    
    # Create and compile model
    model = Model(
        inputs=[cnn_input, lstm_input, sentiment_input],
        outputs=output
    )
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    return model

def train_model(
    X_train_cnn: np.ndarray,
    X_train_lstm: np.ndarray,
    X_train_sent: np.ndarray,
    y_train: np.ndarray
) -> Model:
    """
    Train the hybrid model.
    
    Args:
        X_train_cnn: CNN training data
        X_train_lstm: LSTM training data
        X_train_sent: Sentiment training data
        y_train: Target values
        
    Returns:
        Trained model
    """
    model = build_combined_model(
        (CHART_SIZE[0], CHART_SIZE[1], 3),
        (X_train_lstm.shape[1], X_train_lstm.shape[2])
    )
    
    model.fit(
        [X_train_cnn, X_train_lstm, X_train_sent],
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT
    )
    
    return model