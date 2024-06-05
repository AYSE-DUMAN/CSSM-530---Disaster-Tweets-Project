Disaster Tweet Classification System
Project Overview
This project develops a machine learning model to accurately distinguish between real and fake disaster-related tweets. By leveraging both traditional Natural Language Processing (NLP) techniques and sentiment analysis, the system aims to enhance the reliability of information shared during disaster situations.

Key Features
Real vs. Fake Classification: Classifies tweets as either authentic reports of disasters or as misinformation.
Machine Learning Integration: Uses a variety of machine learning algorithms to optimize classification accuracy.
Sentiment Analysis: Analyzes the emotional tone of tweets to improve classification outcomes.
Topic Modeling: Applies topic modeling to categorize tweets based on content, although its impact on performance is limited.
Implementation
The model employs two main strategies:

Initial Machine Learning Model: Applies conventional NLP techniques (like TF-IDF) and various machine learning algorithms such as Logistic Regression, Random Forest, and XGBoost to a disaster tweet dataset from Kaggle.
Emotion and Topic Analysis: Enhances the model by incorporating emotion analysis from tweets and using topic modeling (Latent Dirichlet Allocation) to understand the contextual topics within tweets.
Performance
Incorporating sentiment analysis markedly improves the model's performance.
The use of topic modeling did not significantly enhance performance.
Algorithms used include:
Logistic Regression
Random Forest
Gradient Boosting
SVM
AdaBoost
Extra Trees
K-Nearest Neighbors
Decision Tree
Bagging Classifier
XGBoost


Notes:

**eda.py**
This Python script includes a set of functions designed to visualize key aspects of a dataset containing tweets related to disasters. These functions utilize the matplotlib and seaborn libraries to create visual representations of the data, focusing on the frequency of keywords and locations within the dataset. Specifically, the script offers functions to plot the top 20 keywords overall, as well as the top 10 keywords specifically associated with disaster and non-disaster tweets. Additionally, it features functions to display the top 15 locations for tweets classified as pertaining to real disasters and those not related to actual disasters. 

**text_processing.py**
This Python script is designed for processing and analyzing text data, specifically tweets, by cleaning text and mapping tweet locations to standardized names. The script first establishes a mapping for common locations to ensure consistency in geographical data, which is crucial for any geospatial analysis. Functions within the script perform several text cleaning tasks such as removing URLs, mentions, hashtags (while keeping the text of the hashtag), and special characters. It also filters out English stopwords to focus on more meaningful words in the text. Additionally, the script identifies and extracts various elements from tweets including hashtags, mentions, links, numeric values, timestamps, and retweets, categorizing each tweet for deeper content analysis. This structured extraction and cleaning allow for a more precise and insightful analysis of the tweet content, making the data ready for further Natural Language Processing (NLP) tasks or data visualization.

**text_features.py **
The textual_feature_generation function aims to extract and calculate various text-related features from a dataset containing tweet data. These features include the length of each tweet in characters, the ratio of capital letters to total characters, the number of words, the number of stopwords, the number of punctuation marks, the number of hashtags, the number of mentions, and the presence of links. These features provide insights into the linguistic characteristics, complexity, and content of the tweets.

**extract_named_entities.py**
The provided function, extract_named_entities.py leverages the nltk library for natural language processing to identify and extract named entities such as locations, people, and organizations from text, particularly useful in analyzing tweet content. The script functions by tokenizing the text, assigning parts of speech, and then using ne_chunk to parse these tokens into a tree structure where subtrees represent recognized named entities. Entities are classified into groups—'GPE' for geographical locations, 'ORGANIZATION' for corporate or group names, and 'PERSON' for individual names. Each identified entity is then added to its respective category within a dictionary, facilitating further analysis of the data based on these extracted details, making it especially valuable in contextual and semantic analysis of social media data.

**LDA model.py**
The function lda_topic_modelling is designed to perform Latent Dirichlet Allocation (LDA) topic modeling on a text dataset, generating topics and identifying key words for each topic. It takes a DataFrame and a specified text column, then uses TF-IDF vectorization to convert the text data into a numerical format suitable for LDA. The function fits an LDA model with a specified number of topics, extracts the top words for each topic, and prints them. Additionally, it assigns a dominant topic to each text entry based on the highest topic probability and adds these dominant topics and their top words as new features to the original DataFrame. This process enhances the dataset with topic-related information, enabling further analysis of the text data.

**sentiment.py**
The provided function, apply_vader_sentiment_analysis, uses the VADER sentiment analyzer to evaluate the sentiment of text data in a specified column of a DataFrame. It first initializes the VADER analyzer and defines an inner function, get_sentiment_scores, which calculates the negative, neutral, positive, and compound sentiment scores for a given text. The main function then applies this inner function to the text column, creating new columns in the DataFrame to store these sentiment scores. As a result, the original DataFrame is returned with additional columns for neg, neu, pos, and compound scores, providing a detailed sentiment analysis of the text data.

