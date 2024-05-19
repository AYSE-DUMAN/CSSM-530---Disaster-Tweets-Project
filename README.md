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
