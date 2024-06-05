# -*- coding: utf-8 -*-
"""
Model4: Performance Metrics for Real and Fake Disaster Tweets with TF-IDF Vectorization
@author: AyseDuman
"""

def Model4(train_full):
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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB

    # Define features and target
    X = train_full[['keyword', 'location', 'text_clean', 'hashtags',
                     'mentions', 'links', 'numbers', 'time', 'rt', 'people', 'organizations', 'text_len', 'ratio_of_capital_letters', 'number_of_words',
                     'number_of_stopwords', 'number_of_punctuation',
                     'number_of_hashtags', 'number_of_mentions', 'number_of_links',
                     'number_of_characters_from_cleaned_text',
                     'number_of_characters_from_text', 'dominant_topic', 'topic_word_0',
                     'neg', 'neu', 'pos', 'compound']]

    y = train_full['target']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define preprocessing for numeric and categorical columns
    numeric_features = ['text_len', 'ratio_of_capital_letters',
                        'number_of_words',
                        'number_of_characters_from_text', 'number_of_characters_from_cleaned_text', 'number_of_stopwords',
                        'number_of_punctuation', 'number_of_hashtags', 'number_of_mentions',
                        'number_of_links', 'dominant_topic', 'neg', 'neu', 'pos', 'compound']

    categorical_features = ['keyword', 'location', 'text_clean', 'hashtags', 'mentions',
                             'links', 'numbers', 'time', 'rt', 'people', 'organizations', 'topic_word_0']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # Create a pipeline for each model using TF-IDF
    tfidf_models = {
        "Logistic Regression": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('clf', LogisticRegression())
        ]),
        "Random Forest": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('clf', RandomForestClassifier())
        ]),
        "Gradient Boosting": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('clf', GradientBoostingClassifier())
        ]),
        "SVM": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('clf', SVC())
        ]),
        "Multinomial Naive Bayes": Pipeline([
            ('tfidf', TfidfVectorizer(max_features=10000)),
            ('clf', MultinomialNB())
        ])
    }

    # Create a pipeline for each model using other preprocessing
    preprocessed_models = {
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

    results_tfidf = []

    for model_name, model_pipeline in tfidf_models.items():
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, output_dict=True)
        precision = report['weighted avg']['precision']
        recall = report['weighted avg']['recall']
        f1 = report['weighted avg']['f1-score']

        results_tfidf.append({
            'Classifier Name': model_name,
            'Class Type': 'Binary',
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

   
# Convert results to DataFrame and display
results_tfidf = pd.DataFrame(results_tfidf)
print(results_tfidf)