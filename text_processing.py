# -*- coding: utf-8 -*-
"""
TEXT PROCESSING 
@author: AyseDuman
"""

import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


location_mapping = {
    'USA': 'United States',
    'United States': 'United States',
    'San Francisco': 'United States',
    'UK': 'United Kingdom',
    'London': 'United Kingdom',
    'Washington, DC': 'United States',
    'Washington, D.C.': 'United States',
    'Chicago, IL': 'United States',
    'New York': 'United States',
    'Los Angeles, CA': 'United States',
    'California, USA': 'United States',
    'Los Angeles': 'United States',
    'Nashville, TN': 'United States',
    'Earth': 'Unknown',
    'California': 'United States',
    'Mumbai': 'India',
    'Toronto': 'Canada',
    'Sacramento, CA': 'United States',
    'New York, NY': 'United States',
    'New York City': 'United States',
    'Denver, Colorado': 'United States',
    'San Francisco, CA ': 'United States',
    'US': 'United States',
    'San Francisco, CA': 'United States',
    'Oklahoma City, OK': 'United States',
    'Atlanta, GA': 'United States',
    'London, UK': 'United Kingdom',
    'ss': 'Unknown',
    'NYC': 'United States',
    'Florida': 'United States',
    'Everywhere': 'Unknown',
    'Worldwide': 'Unknown',
    '304': 'Unknown',
    'London, England': 'United Kingdom',
    'Seattle': 'United States',
    'Texas': 'United States',
    'Chicago': 'United States',
    'Dallas, TX': 'United States',
    'Pennsylvania, USA': 'United States',
    'Manchester': 'United Kingdom',
    'San Diego, CA': 'United States',
    'Morioh, Japan': 'Japan'
}


# Function to map locations in a DataFrame using the provided mapping dictionary
def map_locations(df, location_mapping):
    # Apply the mapping to the location column
    df['location'] = df['location'].map(location_mapping).fillna(df['location'])
    return df


def clean_text(text):
    # Remove line breaks
    text = re.sub(r'\n', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags 
    text = re.sub(r'#(\w+)', r'\1', text)
    # Remove special characters and punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Remove dollar signs and other special symbols
    text = re.sub(r'\$\w*', '', text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    return text


def extract_entities(tweet):
    hashtags = " ".join(re.findall(r"#(\w+)", tweet)) or 'no'
    mentions = " ".join(re.findall(r"@(\w+)", tweet)) or 'no'
    links = 'yes' if re.search(r"https?://\S+", tweet) else 'no'
    numbers = 'yes' if re.search(r'\b\d+\b', tweet) else 'no'
    time = 'yes' if re.search(r'\b\d{1,2}:\d{2}(?::\d{2})?\b', tweet) else 'no'
    rt = 'yes' if re.search(r'\bRT\b', tweet) else 'no'
    return hashtags, mentions, links, numbers, time, rt


def process_text(df):
    df['text_clean'] = df['text'].apply(lambda x: clean_text(x))
    df[['hashtags', 'mentions', 'links', 'numbers', 'time', 'rt']] = df['text'].apply(lambda x: pd.Series(extract_entities(x)))
    return df
