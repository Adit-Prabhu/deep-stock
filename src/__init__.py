"""
Stock Price Prediction Package

This package implements a hybrid deep learning model for stock price prediction
using CNN, LSTM, and sentiment analysis.
"""

from . import data
from . import models
from . import utils

__version__ = '1.0.0'
__author__ = 'Adit Prabhu'
__email__ = 'adprabh2@asu.edu'

# Submodule imports for convenient access
from .data.stock_data import get_stock_data, create_candlestick_chart
from .data.image_data import load_images
from .data.sentiment_data import get_financial_news_sentiment
from .models.hybrid_model import build_combined_model, train_model
from .utils.preprocessing import prepare_time_series_data, align_data_lengths

__all__ = [
    'get_stock_data',
    'create_candlestick_chart',
    'load_images',
    'get_financial_news_sentiment',
    'build_combined_model',
    'train_model',
    'prepare_time_series_data',
    'align_data_lengths'
]