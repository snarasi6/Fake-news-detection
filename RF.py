 #Random Forest approach - Fake-news detection - ALDA project - Group 3
import pandas as pd # data processing
import re
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

news_data = pd.read_csv("news_dataset.csv")
news_data['label'] = news_data['category'].apply(lambda x: 0 if x=='real' else 1)

#Removing all the unwanted nonwords, numbers, articles removed using refineWords
def refineWords(s):
    letters_only = re.sub("[^a-zA-Z]", " ", s)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return( " ".join( meaningful_words ))

news_data["content"].fillna(" ", inplace=True)
news_data["content"] = news_data["content"].apply(refineWords)
news_data["title"].fillna(" ", inplace=True)
news_data["title"] = news_data["title"].apply(refineWords)
news_data["publication"].fillna(" ", inplace=True)
news_data["publication"] = news_data["publication"].apply(refineWords)
print("The shape of the dataset after processing:")
print(news_data.shape)
X_train, X_test, y_train, y_test = train_test_split(news_data['content'], news_data['label'], test_size=0.2, random_state=1)

#CountVectorizer creates a vector of word counts for each of the content/title to form the matrix
cv = CountVectorizer(strip_accents='ascii', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b', lowercase=True, stop_words='english')
X_train_cv = cv.fit_transform(X_train)
X_test_cv = cv.transform(X_test)
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(X_train_cv, y_train)
prediction = my_forest.predict(X_test_cv)
print('Accuracy score: ', accuracy_score(y_test, prediction))
print('Precision score: ', precision_score(y_test, prediction))
print('Recall score: ', recall_score(y_test, prediction))
print('RoC_AuC Value:', roc_auc_score(y_test, prediction))
fpr, tpr, thresholds = roc_curve(y_test, prediction)
plt.plot(fpr, tpr)
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
figsize = (10,7)
fontsize=14
labels = ['Real News', 'Fake News']
cm = confusion_matrix(y_test, prediction)
df_cm = pd.DataFrame(cm, index=labels, columns=labels)
fig = plt.figure(figsize=figsize)
heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
bottom, top = heatmap.get_ylim()
heatmap.set_ylim(bottom, top)
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.savefig('a.png')
plt.show()
