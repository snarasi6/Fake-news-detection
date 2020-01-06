#Preprocessing of datasets for fake-news detection - ALDA project - Group 3
import numpy as np # linear algebra
import pandas as pd # data processing
import os

fake_news = pd.read_csv("/Users/akhilesh/Desktop/fake.csv")
real_news = pd.read_csv("/Users/akhilesh/Desktop/real_news.csv")
#Setting labels for the dataset with the real news
real_news_new = real_news[['title', 'content', 'publication']]
real_news_new['category'] = 'real'
#Setting the labels for dataset with the fake news
fake_news_new = fake_news[['title', 'text','site_url']]
fake_news_new['category'] = 'fake'
#Removing domain name from the site_url to form the publication for fake news
site_url_new = fake_news_new['site_url']
site_url_new = [x.split('.',1)[0] for x in site_url_new]
fake_news_new['site_url'] = site_url_new
#Creating the new labels
newlabels = ['title', 'content', 'publication', 'category']
real_news_new.columns = newlabels
fake_news_new.columns = newlabels
#Combining the 2 datasets to form the complete dataset for analysis
frames = [fake_news_new, real_news_new]
news_dataset = pd.concat(frames)
#print(news_dataset.head(5))
#Save the dataset as a CSV file
news_dataset.to_csv('/Users/akhilesh/Desktop/news_dataset.csv', encoding='utf-8')
#Get dimensions of the dataset
print(news_dataset.shape)
