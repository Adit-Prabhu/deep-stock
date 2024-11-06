"""Main script for running the stock prediction model."""

import os
import numpy as np
from sklearn.model_selection import train_test_split
from src.data.stock_data import get_stock_data, create_candlestick_chart
from src.data.image_data import load_images
from src.data.sentiment_data import get_financial_news_sentiment
from src.utils.preprocessing import prepare_time_series_data, align_data_lengths
from src.models.hybrid_model import train_model
from config import (
    CHART_DIR,
    TEST_SIZE,
    RANDOM_STATE
)

def main():
    """Main function to run the stock prediction pipeline."""
    try:
        # Data Collection
        print("Collecting stock data...")
        stock_data = get_stock_data('AAPL', '2020-01-01', '2023-01-01')
        
        # Generate Charts
        print("Generating charts...")
        create_candlestick_chart(stock_data, CHART_DIR)
        
        # Prepare Different Data Types
        print("Preparing data...")
        # Time series data
        X_lstm, y, scaler = prepare_time_series_data(stock_data)
        
        # Image data
        X_cnn = load_images(CHART_DIR)
        
        # Sentiment data (replace with actual news source)
        news_url = "https://finance.yahoo.com/quote/AAPL/news"
        sentiment_scores = []
        for _ in range(len(y)):  # In practice, collect real sentiment data
            sentiment = get_financial_news_sentiment(news_url)
            sentiment_scores.append([sentiment])
        sentiment_data = np.array(sentiment_scores)
        
        # Align Data Lengths
        X_cnn, X_lstm, sentiment_data, y = align_data_lengths(
            X_cnn, X_lstm, sentiment_data, y
        )
        
        # Split Data
        print("Splitting data...")
        (X_train_cnn, X_test_cnn,
         X_train_lstm, X_test_lstm,
         X_train_sent, X_test_sent,
         y_train, y_test) = train_test_split(
            X_cnn, X_lstm, sentiment_data, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )
        
        # Train Model
        print("Training model...")
        model = train_model(
            X_train_cnn,
            X_train_lstm,
            X_train_sent,
            y_train
        )
        
        # Evaluate Model
        print("Evaluating model...")
        loss, mae = model.evaluate(
            [X_test_cnn, X_test_lstm, X_test_sent],
            y_test,
            verbose=1
        )
        print(f"Test Loss: {loss:.4f}")
        print(f"Test MAE: {mae:.4f}")
        
        # Save Model
        print("Saving model...")
        model.save('deep-stock-predictor.keras')
        print("Model saved successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()