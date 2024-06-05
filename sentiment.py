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

# Function to apply VADER sentiment analysis on a DataFrame
def apply_vader_sentiment_analysis(df, text_column):
    """
    Applies VADER sentiment analysis to a specified text column in a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing the text data.
        text_column (str): The name of the column in the DataFrame that contains the text to analyze.

    Returns:
        pd.DataFrame: The original DataFrame with additional columns for sentiment scores:
                      'neg', 'neu', 'pos', and 'compound'.
    """
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Function to get sentiment scores for a given text
    def get_sentiment_scores(text):
        """
        Computes sentiment scores for a given text using VADER sentiment analysis.

        Args:
            text (str): The text to analyze.

        Returns:
            tuple: A tuple containing the negative, neutral, positive, and compound sentiment scores.
        """
        scores = sid.polarity_scores(text)
        return scores['neg'], scores['neu'], scores['pos'], scores['compound']

    # Apply sentiment analysis to the specified text column and create new columns for the scores
    df['neg'], df['neu'], df['pos'], df['compound'] = zip(*df[text_column].map(get_sentiment_scores))

    return df

# Example usage:
# train_full = apply_vader_sentiment_analysis(train_full, 'text_clean')
