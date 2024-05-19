# -*- coding: utf-8 -*-
"""
@author: AyseDuman
"""
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

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