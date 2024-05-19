# -*- coding: utf-8 -*-
"""


@author: AyseDuman
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

def lda_topic_modelling(data, text_column, n_topics=10, n_top_words=1, max_features=5000):
    # TF-IDF
    vectorizer = TfidfVectorizer(max_features=max_features)
    X = vectorizer.fit_transform(data[text_column])

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    # Function to get top words for each topic
    def get_top_words(model, feature_names, n_top_words):
        top_words = {}
        for topic_idx, topic in enumerate(model.components_):
            top_words[f"topic_{topic_idx}"] = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        return top_words

    # Get top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    top_words = get_top_words(lda, feature_names, n_top_words)

    print("Top words for each topic:")
    for topic, words in top_words.items():
        print(f"{topic}: {', '.join(words)}")

    # Assign topics to tweets based on the highest topic probability
    topic_distribution = lda.transform(X)
    data['dominant_topic'] = topic_distribution.argmax(axis=1)

    # Add top words of the dominant topic as features
    for i in range(n_top_words):
        data[f'topic_word_{i}'] = data['dominant_topic'].apply(lambda topic: top_words[f'topic_{topic}'][i])

    return data

