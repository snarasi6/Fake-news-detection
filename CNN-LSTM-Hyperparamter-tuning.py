import re
import pandas as pd
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM,Dense, Dropout, Embedding, Flatten, Conv1D, MaxPooling1D, Activation
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
# from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import talos as ta
from keras.optimizers import Adam, RMSprop
from keras.activations import relu, elu, sigmoid
from keras.losses import logcosh, binary_crossentropy
from talos.model.normalizers import lr_normalizer

# Pre-processing
# Use pd to read from json file and get dataframe
# Generate headlines list and label list from dataframe
# Remove nltk stopwords and string punctuations
# X: headlines Y: label
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
Y = Y.values.reshape(-1, 1)

x_train, y_train, x_test, y_test = train_test_split(X, Y, test_size = 0.2)

p = {'lr': [0.5, 5, 10],
     'max_words': [40000],
     'max_length': [20, 50, 100, 150],
     'batch_size': [10, 30, 100],
     'epochs': [150],
     'dropout': [0, 0.2, 0.5],
     'optimizer': [Adam, RMSprop],
     'losses': [logcosh, binary_crossentropy],
     'activation': [relu, elu],
     'last_activation': [sigmoid]}


def fake_news_model(x_train, y_train, x_val, y_val, params):

    t = Tokenizer(num_words=params['max_words'])
    t.fit_on_texts(x_train)
    train_sequence = t.texts_to_sequences(x_train)
    train_padded = pad_sequences(train_sequence, maxlen=params['max_length'])
    
    # next we can build the model exactly like we would normally do it
    model = Sequential()
    model.add(Embedding(params['max_words'], 64, input_length=params['max_length']))
    model.add(Dropout(params['dropout']))    
    model.add(Conv1D(64, 5, activation=params['activation']))
    model.add(MaxPooling1D(pool_size=4))
    model.add(LSTM(100))
    model.add(Dense(1, activation=params['activation']))
    model.compile(loss=params['losses'],
                  optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(train_padded, y_train, 
                        batch_size=params['batch_size'],
                        epochs=150,
                        validation_split=0.1,
                        callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.00001)],
                        verbose=2)
    
    return history, model
    
t = ta.Scan(x=X,
            y=Y,
            model=fake_news_model,
            fraction_limit=0.01, 
            params=p,
            experiment_name='fake_news_1')
            
reporting = ta.Reporting(t)
report = reporting.data
report.to_csv('report_talos.txt', sep = '\t')
