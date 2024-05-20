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

    
    
def train_and_evaluate_models(df, features, target, models, test_size=0.2, random_state=42):


    # Define features and target
    X = df[features]
    y = df[target]

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define preprocessing for numeric and categorical columns
    numeric_features = ['text_len', 'ratio_of_capital_letters',
                        'number_of_words', 'number_of_characters_from_cleaned_text',
                        'number_of_characters_from_text', 'number_of_stopwords',
                        'number_of_punctuation', 'number_of_hashtags', 'number_of_mentions',
                        'number_of_links', 'neg', 'neu', 'pos', 'compound']

    categorical_features = ['keyword', 'final_location', 'text_clean', 'hashtags', 'mentions',
                            'links', 'people', 'organizations', 'dominant_topic',
                            'topic_word_0']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    results = []

    # Train and evaluate each model
    for model_name, model_pipeline in models.items():
        # Create a pipeline with preprocessing and the model
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('clf', model_pipeline)
        ])

        # Train the model
        pipeline.fit(X_train, y_train)

        # Predict on validation set
        y_pred = pipeline.predict(X_val)

        # Evaluate the model
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        # Append results to the list
        results.append({
            'Classifier Name': model_name,
            'Class Type': 'Binary',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    # Convert results to DataFrame and return
    results_df = pd.DataFrame(results)
    return results_df

# Usage example

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "AdaBoost": AdaBoostClassifier(),
    "Extra Trees": ExtraTreesClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Bagging Classifier": BaggingClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Define features and target
features = ['keyword', 'location', 'text', 'text_clean', 'hashtags',
            'mentions', 'links', 'numbers', 'time', 'rt', 'text_len', 'ratio_of_capital_letters',
            'number_of_words', 'number_of_characters_from_cleaned_text',
            'number_of_characters_from_text', 'number_of_stopwords',
            'number_of_punctuation', 'number_of_hashtags', 'number_of_mentions',
            'number_of_links', 'dominant_topic', 'topic_word_0', 'neg', 'neu',
            'pos', 'compound']

target = 'target'

# Train and evaluate models
results_df = train_and_evaluate_models(train_full, features, target, models)
print(results_df)
