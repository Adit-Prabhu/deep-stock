"""Module for collecting and processing stock market data."""

import yfinance as yf
import pandas as pd
from typing import Tuple, Optional
import matplotlib.pyplot as plt
import os
from ...config import CHART_DIR, WINDOW_SIZE

def get_stock_data(
    ticker: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Download stock data for a given ticker and date range.
    
    Args:
        ticker: Stock symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame containing stock data
    """
    data = yf.download(ticker, start=start_date, end=end_date)
    data['Date'] = data.index
    return data

def create_candlestick_chart(
    data: pd.DataFrame,
    save_dir: str = CHART_DIR
) -> None:
    """
    Create and save candlestick charts for the given stock data.
    
    Args:
        data: DataFrame containing stock data
        save_dir: Directory to save generated charts
    """
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(WINDOW_SIZE, len(data)):
        sub_data = data.iloc[i-WINDOW_SIZE:i]
        plt.figure(figsize=(6, 3))
        plt.plot(sub_data['Close'], label='Close Price')
        plt.title('Stock Price')
        plt.axis('off')
        filepath = os.path.join(save_dir, f'chart_{i}.png')
        plt.savefig(filepath, bbox_inches='tight', pad_inches=0)
        plt.close()