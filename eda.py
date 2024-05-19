# -*- coding: utf-8 -*-
"""

@author: AyseDuman
"""

import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_keywords(train):
    plt.figure(figsize=(9,6))
    sns.countplot(y=train.keyword, order=train.keyword.value_counts().iloc[:20].index)
    plt.title("Top 20 keywords")
    plt.show()

def plot_top_keywords_disaster(train):
    train_d_kw = train[train.target==1].keyword.value_counts().head(10)
    sns.barplot(train_d_kw, train_d_kw.index)
    plt.title("Top 10 keywords in disaster twitter dataset")
    plt.show()

def plot_top_keywords_non_disaster(train):
    train_nd_kw = train[train.target==0].keyword.value_counts().head(10)
    sns.barplot(train_nd_kw, train_nd_kw.index)
    plt.title("Top 10 keywords in non-disaster twitter dataset")
    plt.show()

def plot_top_locations_disaster(train):
    train_d = train[train.target==1]
    plt.figure(figsize=(9,6))
    sns.countplot(y=train_d.location, order=train_d.location.value_counts().iloc[:15].index)
    plt.title('Top 15 locations in real disaster dataset')
    plt.show()

def plot_top_locations_non_disaster(train):
    train_n = train[train.target==0]
    plt.figure(figsize=(9,6))
    sns.countplot(y=train_n.location, order=train_n.location.value_counts().iloc[:15].index)
    plt.title('Top 15 locations in non-real disaster dataset')
    plt.show()


