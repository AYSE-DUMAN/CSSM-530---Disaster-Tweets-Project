# -*- coding: utf-8 -*-
"""
Performance Metrics Without Using Topic Keywords  for Real and Fake Disaster Tweets

@author: AyseDuman
"""

def Model3(train_full):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import classification_report, accuracy_score
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    # Define features and target
    X = train_full[['keyword', 'location', 'text_clean', 'hashtags',
                     'mentions', 'links', 'numbers', 'time', 'rt', 'people', 'organizations', 'text_len', 'ratio_of_capital_letters', 'number_of_words',
                     'number_of_stopwords', 'number_of_punctuation',
                     'number_of_hashtags', 'number_of_mentions', 'number_of_links',
                     'number_of_characters_from_cleaned_text',
                     'number_of_characters_from_text',
                     'neg', 'neu', 'pos', 'compound']]

    y = train_full['target']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing for numeric and categorical columns
    numeric_features = ['text_len', 'ratio_of_capital_letters',
                        'number_of_words',
                        'number_of_characters_from_text', 'number_of_characters_from_cleaned_text', 'number_of_stopwords',
                        'number_of_punctuation', 'number_of_hashtags', 'number_of_mentions',
                        'number_of_links', 'neg', 'neu', 'pos', 'compound']

    categorical_features = ['keyword', 'location', 'text_clean', 'hashtags', 'mentions',
                             'links', 'numbers', 'time', 'rt', 'people', 'organizations']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create a pipeline for each model
    models = {
        "Logistic Regression": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', LogisticRegression())
        ]),
        "Random Forest": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', RandomForestClassifier())
        ]),
        "Gradient Boosting": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', GradientBoostingClassifier())
        ]),
        "SVM": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', SVC())
        ]),
        "AdaBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', AdaBoostClassifier())
        ]),
        "Extra Trees": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', ExtraTreesClassifier())
        ]),
        "K-Nearest Neighbors": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', KNeighborsClassifier())
        ]),
        "Decision Tree": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', DecisionTreeClassifier())
        ]),
        "Bagging Classifier": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', BaggingClassifier())
        ]),
        "XGBoost": Pipeline([
            ('preprocessor', preprocessor),
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
        ])
    }

    results3 = []

    for model_name, model_pipeline in models.items():
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        results3.append({
            'Classifier Name': model_name,
            'Class Type': 'Binary',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    # Convert results to DataFrame and return
    results_df3 = pd.DataFrame(results3)
    return results_df3
