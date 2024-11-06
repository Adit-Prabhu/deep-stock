# Stock Price Prediction with Deep Learning

This project implements a hybrid deep learning model for stock price prediction using CNN, LSTM, and sentiment analysis. The model combines technical analysis through price charts, time series data, and market sentiment to make predictions.

## Project Structure
```
deep-stock/
├── README.md
├── requirements.txt
├── config.py
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── stock_data.py
│   │   ├── image_data.py
│   │   └── sentiment_data.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn.py
│   │   ├── lstm.py
│   │   └── hybrid_model.py
│   └── utils/
│       ├── __init__.py
│       └── preprocessing.py
└── main.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.data.stock_data import get_stock_data
from src.models.hybrid_model import train_model

# Get stock data
data = get_stock_data('AAPL', '2020-01-01', '2023-01-01')

# Train model
model = train_model(data)
```

## Features
- Stock data collection using yfinance
- Technical analysis through candlestick charts
- Sentiment analysis from financial news
- Hybrid deep learning model combining CNN and LSTM
- Real-time prediction capabilities

## Requirements
See requirements.txt for full list of dependencies.
