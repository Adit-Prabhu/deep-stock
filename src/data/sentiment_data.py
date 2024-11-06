"""Module for collecting and analyzing sentiment data from financial news."""

import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import numpy as np
from typing import List, Optional, Union
import nltk

# Ensure nltk is set up
nltk.download('punkt')

def get_sentiment_score(news_text: str) -> float:
    """
    Calculate sentiment score for a given text.
    
    Args:
        news_text: Text to analyze
        
    Returns:
        Sentiment polarity score between -1 and 1
    """
    analysis = TextBlob(news_text)
    return analysis.sentiment.polarity

def get_financial_news_sentiment(url: str) -> float:
    """
    Analyze sentiment from financial news headlines.
    
    Args:
        url: URL of the financial news page
        
    Returns:
        Average sentiment score of all headlines
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        headlines = [h.get_text() for h in soup.find_all('h2')]
        sentiments = [get_sentiment_score(h) for h in headlines]
        return np.mean(sentiments) if sentiments else 0.0
    except Exception as e:
        print(f"Error fetching news: {str(e)}")
        return 0.0