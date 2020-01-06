#Naive Bayes approach - Fake-news detection - ALDA project - Group 3
import numpy as np # linear algebra
import pandas as pd # data processing
import os
import re
from nltk.corpus import stopwords
# import word2vec
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dense, Dropout, Embedding
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
# from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

news_data = pd.read_csv("/Users/akhilesh/Desktop/news_dataset.csv")
# print(news_data.head(5))
news_data['label'] = news_data['category'].apply(lambda x: 0 if x=='real' else 1)
# print(news_data.head(5))

#Removing all the unwanted nonwords, numbers, articles removed using refineWords
def refineWords(s):
    letters_only = re.sub("[^a-zA-Z]", " ", s)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    #print( " ".join( meaningful_words ))
    return( " ".join( meaningful_words ))

news_data["content"].fillna(" ", inplace=True)
news_data["content"] = news_data["content"].apply(refineWords)
news_data["title"].fillna(" ", inplace=True)
news_data["title"] = news_data["title"].apply(refineWords)
news_data["publication"].fillna(" ", inplace=True)
news_data["publication"] = news_data["publication"].apply(refineWords)
news_data.to_csv('/Users/akhilesh/Desktop/news_dataset.csv', encoding='utf-8')
print("The shape of the dataset after processing:")
print(news_data.shape)
