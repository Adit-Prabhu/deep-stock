"""Configuration parameters for the stock prediction model."""

# Data Collection
WINDOW_SIZE = 30
CHART_SIZE = (64, 64)
CHART_DIR = './charts'

# Model Parameters
CNN_FILTERS = [32, 64]
CNN_KERNEL_SIZE = (3, 3)
LSTM_UNITS = [50, 50]
DENSE_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.5

# Training Parameters
BATCH_SIZE = 32
EPOCHS = 10
VALIDATION_SPLIT = 0.2
TEST_SIZE = 0.2
RANDOM_STATE = 42