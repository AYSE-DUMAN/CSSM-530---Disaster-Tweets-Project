# -*- coding: utf-8 -*-
"""

@author: AyseDuman
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd

def evaluate_tfidf_models(df, text_column, target_column, models, max_features=10000, test_size=0.2, random_state=42):
    
    X_train, X_val, y_train, y_val = train_test_split(df[text_column], df[target_column], test_size=test_size, random_state=random_state)
    
    results = []

    # Train and evaluate each model
    for model_name, model_pipeline in models.items():

        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=max_features)),
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

        results.append({
            'Classifier Name': model_name,
            'Class Type': 'Binary',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    results_df = pd.DataFrame(results)
    return results_df

# Define TF-IDF models
tfidf_models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "SVM": SVC(),
    "Multinomial Naive Bayes": MultinomialNB()
}

# Usage example
results_tfidf = evaluate_tfidf_models(train_full, 'text_clean', 'target', tfidf_models)
print(results_tfidf)
