# -*- coding: utf-8 -*-
"""
@author: AyseDuman
"""

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def train_test_split_with_preprocessing(data, target, numeric_features, categorical_features, test_size=0.2, random_state=42):
    """
    Perform train-test split and apply preprocessing to the data.

    Args:
    - data (DataFrame): The input DataFrame containing the features.
    - target (str): The name of the target column.
    - numeric_features (list): List of column names for numeric features.
    - categorical_features (list): List of column names for categorical features.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
    - X_train (DataFrame): The training data with preprocessed features.
    - X_val (DataFrame): The validation data with preprocessed features.
    - y_train (Series): The training target.
    - y_val (Series): The validation target.
    - preprocessor (ColumnTransformer): The fitted preprocessor to be used for transforming new data.
    """
    X = data.drop(columns=[target])
    y = data[target]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define preprocessing for numeric and categorical columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Fit and transform the preprocessor on the training data
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)

    return X_train, X_val, y_train, y_val, preprocessor


# Define the numeric and categorical features
numeric_features = ['text_len', 'ratio_of_capital_letters', 'number_of_words',
                     'number_of_characters_from_text', 'number_of_stopwords',
                     'number_of_punctuation', 'number_of_hashtags', 'number_of_mentions',
                     'number_of_links', 'dominant_topic', 'neg', 'neu', 'pos', 'compound']

categorical_features = ['keyword', 'location', 'text_clean', 'hashtags', 'mentions', 
                         'links', 'numbers', 'time', 'rt', 'people', 'organizations', 'topic_word_0']

# Apply the function to get the train-test split and preprocessor
#X_train, X_val, y_train, y_val, preprocessor = train_test_split_with_preprocessing(train_full, 'target', numeric_features, categorical_features)
