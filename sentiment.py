# -*- coding: utf-8 -*-
"""

@author: AyseDuman
"""
# Import necessary libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# Install the VADER sentiment analysis library
!pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host=files.pythonhosted.org vaderSentiment
# Download the VADER lexicon
nltk.download('vader_lexicon')

# Function to install and use VADER sentiment analysis on a DataFrame
def apply_vader_sentiment_analysis(df, text_column):

    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()
    # Function to get sentiment scores
    def get_sentiment_scores(text):
        scores = sid.polarity_scores(text)
        return scores['neg'], scores['neu'], scores['pos'], scores['compound']

    # Apply sentiment analysis to the specified text column and create new columns for the scores
    df['neg'], df['neu'], df['pos'], df['compound'] = zip(*df[text_column].map(get_sentiment_scores))

    return df