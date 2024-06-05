

# -*- coding: utf-8 -*-
"""
text_len: Calculates the length of each tweet in characters, representing the total number of characters in the cleaned text of the tweet.
ratio_of_capital_letters: Computes the ratio of capital letters to the total number of characters in the cleaned text, indicating the proportion of capital letters in the tweet.
number_of_words: Counts the number of words in the cleaned text of each tweet.
number_of_characters_from_cleaned_text: Counts the total number of characters in the cleaned text of each tweet.
number_of_characters_from_text: Counts the total number of characters in the original (uncleaned) text of each tweet.
number_of_stopwords: Calculates the number of stopwords (common words like "the", "is", "and", etc.) in each tweet, which can provide insights into the complexity of the text.
number_of_punctuation: Counts the number of punctuation marks in the cleaned text of each tweet, which can indicate the level of emphasis or emotion in the tweet.
number_of_hashtags: Counts the number of hashtags in each tweet, providing information about the topics or themes discussed in the tweets.
number_of_mentions: Counts the number of mentions (usernames preceded by "@") in each tweet, indicating the level of interaction or engagement with other users.
number_of_links: Flags whether a tweet contains a link, represented as 1 if a link is present and 0 otherwise.

"""


import string
import pandas as pd
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words('english'))


def textual_feature_generation(df):
    df["text_len"] = df["text_clean"].astype(str).apply(len)  # length of tweet
    df["ratio_of_capital_letters"] = df["text_clean"].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / len(x) if len(x) > 0 else 0)
    df['number_of_words'] = df["text_clean"].apply(lambda x: len(str(x).split()))
    df["number_of_characters_from_cleaned_text"] = df["text_clean"].apply(lambda x: len(str(x)))
    df["number_of_characters_from_text"] = df["text"].apply(lambda x: len(str(x)))
    df["number_of_stopwords"] = df["text"].apply(lambda x: len([w for w in str(x).lower().split() if w in STOPWORDS]))
    df["number_of_punctuation"] = df["text_clean"].apply(lambda x: len([c for c in str(x) if c in string.punctuation]))
    df["number_of_hashtags"] = df["hashtags"].apply(lambda x: len(str(x).split()) if x != "no" else 0)
    df["number_of_mentions"] = df["mentions"].apply(lambda x: len(str(x).split()) if x != "no" else 0)
    df["number_of_links"] = df["links"].apply(lambda x: 1 if x == 'yes' else 0)
    
    return df


def remove_words(text, words_to_remove):
    """
    Remove specified words from the text.

    Args:
    - text (str): The input text.
    - words_to_remove (list): A list of words to be removed.

    Returns:
    - str: The text with specified words removed.
    """
    return ' '.join([word for word in text.split() if word.lower() not in words_to_remove])


# Define the words to remove
#words_to_remove = ['im', 'mh370', 'via']

# Apply the function to the 'text_clean' column of train_full DataFrame
#train_full['text_clean'] = train_full['text_clean'].apply(lambda x: remove_words(x, words_to_remove))


import pandas as pd
import numpy as np

def update_location_based_on_locations(df):
    """
    Update the 'location' column based on the 'locations' column in a DataFrame.
    
    Args:
    df (pd.DataFrame): Input DataFrame with 'location' and 'locations' columns.
    
    Returns:
    pd.DataFrame: DataFrame with updated 'location' column.
    """
    def update_location(row):
        """
        Update the 'location' value for a single row based on the 'locations' value.
        
        Args:
        row (pd.Series): A single row of the DataFrame.
        
        Returns:
        str: Updated 'location' value for the row.
        """
        if pd.isna(row['location']) and len(row['locations']) > 0:
            # Join the list of locations into a single string separated by commas
            return ', '.join(row['locations'])
        return row['location']
    
    # Apply the update_location function to each row in the DataFrame
    df['location'] = df.apply(update_location, axis=1)
    return df


def update_people(row):
    """
    Update the 'people' value for a single row based on the 'people' value.
    
    Args:
    row (pd.Series): A single row of the DataFrame.
    
    Returns:
    str: Updated 'people' value for the row.
    """
    if isinstance(row['people'], list) and len(row['people']) > 0:
        # Join the list of people into a single string separated by commas
        return ', '.join(row['people'])
    elif isinstance(row['people'], list) and len(row['people']) == 0:
        return np.nan
    return row['people']

def update_organizations(row):
    """
    Update the 'organizations' value for a single row based on the 'organizations' value.
    
    Args:
    row (pd.Series): A single row of the DataFrame.
    
    Returns:
    str: Updated 'organizations' value for the row.
    """
    if isinstance(row['organizations'], list) and len(row['organizations']) > 0:
        # Join the list of organizations into a single string separated by commas
        return ', '.join(row['organizations'])
    elif isinstance(row['organizations'], list) and len(row['organizations']) == 0:
        return np.nan
    return row['organizations']



# Apply the update_organizations function to each row in the DataFrame
train_full['organizations'] = train_full.apply(update_organizations, axis=1)


train_full['people'] = train_full.apply(update_people, axis=1)



# Apply the update_organizations function to each row in the DataFrame
# train_full['organizations'] = train_full.apply(update_organizations, axis=1)

#train_full['people'] = train_full.apply(update_people, axis=1)

#train_full = update_location_based_on_locations(train)


