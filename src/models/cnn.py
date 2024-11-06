"""CNN model for processing chart images."""

from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from typing import Tuple
from ...config import CNN_FILTERS, CNN_KERNEL_SIZE, DENSE_UNITS, DROPOUT_RATE

def build_cnn_model(input_shape: Tuple[int, int, int]) -> Tuple[Input, Dense]:
    """
    Build CNN model for processing chart images.
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Tuple containing:
        - Input layer
        - Output tensor
    """
    input_layer = Input(shape=input_shape)
    
    # First convolutional block
    x = Conv2D(CNN_FILTERS[0], CNN_KERNEL_SIZE,
               activation='relu', padding='same')(input_layer)
    
    # Second convolutional block
    x = Conv2D(CNN_FILTERS[1], CNN_KERNEL_SIZE,
               activation='relu', padding='same')(x)
    
    # Flatten and dense layers
    x = Flatten()(x)
    x = Dense(DENSE_UNITS[0], activation='relu')(x)
    x = Dropout(DROPOUT_RATE)(x)
    
    return input_layer, x