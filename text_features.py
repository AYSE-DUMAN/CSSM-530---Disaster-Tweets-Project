# -*- coding: utf-8 -*-
"""

@author: AyseDuman
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