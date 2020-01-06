import re
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dense, Dropout, Embedding
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

news_data = pd.read_csv('news_dataset.csv')
news_data['label'] = news_data['category'].apply(lambda x: 0 if x=='real' else 1)

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
print("The shape of the dataset after processing:")
print(news_data.shape)


X = news_data.content
Y = news_data.label
Y = Y.reshape(-1, 1)




seed = 0
k_fold = KFold(n_splits=10, shuffle=True, random_state=seed)
train_acc = []
val_acc = []
test_acc = []
prec = []
recall = []
roc = []
fpr = []
tpr = []
roc_auc = 0
for train, test in k_fold.split(X, Y):
    max_words = 40000
    max_length = 150
    t = Tokenizer(num_words=max_words)
    t.fit_on_texts(X[train])
    train_sequence = t.texts_to_sequences(X[train])
    train_padded = pad_sequences(train_sequence, maxlen=max_length)
    model = Sequential()
    model.add(Embedding(max_words, 64, input_length=max_length))
    model.add(Dropout(0.2))
    model.add(LSTM(64))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
    model.summary()
    histroy = model.fit(train_padded, Y[train], batch_size=100, epochs=150, validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)])
#    histroy = model.fit(train_padded, Y[train], batch_size=100, epochs=25, validation_split=0.1)
    train_acc.append(histroy.history['acc'][-1])
    val_acc.append(histroy.history['val_acc'][-1])
    test_sequence = t.texts_to_sequences(X[test])
    test_padded = pad_sequences(test_sequence, maxlen=max_length)
    test_scores = model.evaluate(test_padded, Y[test])
    predictions = model.predict_classes(test_padded)
    predictions = predictions[:,0]
    prec.append(precision_score(Y[test], predictions))
    recall.append(recall_score(Y[test], predictions))
    roc.append(roc_auc_score(Y[test], predictions))
    fpr, tpr, thresholds = roc_curve(Y[test], predictions)
    roc_auc = auc(fpr, tpr)
    test_acc.append(test_scores[1])

print(train_acc)
print(val_acc)
print(test_acc)
print(prec)
print(recall)
print(roc)
print("training accuracy:", np.mean(train_acc))
print("validation accuracy:", np.mean(val_acc))
print("test accuracy:", np.mean(test_acc))
print("Precision:", np.mean(prec))
print("Recall:", np.mean(recall))
print("ROC AUC:", np.mean(roc))

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
