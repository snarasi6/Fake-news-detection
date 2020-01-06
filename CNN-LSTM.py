  #CNN - LSTM approach - Fake-news detection - ALDA project - Group 3
import numpy as np # linear algebra
import pandas as pd # data processing
import re
from nltk.corpus import stopwords
from keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

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

X = news_data.content
Y = news_data.label
seed = 0
k_fold = KFold(n_splits=5, shuffle=True, random_state=seed)
train_acc = []
val_acc = []
test_acc = []
predictions = []
prec = []
recall = []
roc = []
for train, test in k_fold.split(X, Y):
    max_words = 40000
    max_length = 100
    t = Tokenizer(num_words=max_words)
    t.fit_on_texts(X[train])
    train_sequence = t.texts_to_sequences(X[train])
    train_padded = pad_sequences(train_sequence, maxlen=max_length)
    model = Sequential()
    model.add(Embedding(max_words, 100, input_length=max_length))
    model.add(Dropout(0))
    model.add(Conv1D(64, 5, activation='elu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='logcosh', optimizer='adam', metrics=['accuracy'])
    model.summary()
    histroy = model.fit(train_padded, Y[train], batch_size=1000, epochs=1, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)], validation_split=0.1)
    train_acc.append(histroy.history['accuracy'][-1])
    val_acc.append(histroy.history['val_accuracy'][-1])
    test_sequence = t.texts_to_sequences(X[test])
    test_padded = pad_sequences(test_sequence, maxlen=max_length)
    test_scores = model.evaluate(test_padded, Y[test])
    test_acc.append(test_scores[1])
    
    predictions = model.predict_classes(test_padded)
    predictions = predictions[:,0]
    prec.append(precision_score(Y[test], predictions))
    recall.append(recall_score(Y[test], predictions))
    roc.append(roc_auc_score(Y[test], predictions))

print(train_acc)
print(val_acc)
print(test_acc)
print("training accuracy:", np.mean(train_acc))
print("validation accuracy:", np.mean(val_acc))
print("test accuracy:", np.mean(test_acc))
print("Precision:", np.mean(prec))
print("Recall:", np.mean(recall))
print("ROC AUC:", np.mean(roc))
fpr, tpr, thresholds = roc_curve(Y[test], predictions)
plt.plot(fpr, tpr)
plt.savefig('a.png')
plt.show()
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

SVG(model_to_dot(model).create(prog='dot', format='svg'))
