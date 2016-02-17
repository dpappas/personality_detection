__author__ = 'Dimitris'


from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.regularizers import l1, activity_l1

def create_simple_LSTM (LSTM_size, Dense_size, embeddings, max_input_length, is_trainable):
    D = embeddings.shape[-1]
    out_dim = 5
    model = Sequential()
    model.add(Embedding(input_dim = embeddings.shape[0],
                        output_dim=D,
                        weights=[embeddings],
                        trainable=is_trainable,
                        input_length = max_input_length))
    model.add(LSTM(LSTM_size, activation = 'sigmoid'))
    model.add(Dense(Dense_size, activation = 'sigmoid',W_regularizer=l1(0.2), activity_regularizer=activity_l1(0.2)))
    model.add(Dense(out_dim, activation = 'linear',W_regularizer=l1(0.2), activity_regularizer=activity_l1(0.2)))
    model.compile(loss='rmse', optimizer='sgd') # kalutera leei rmsprop o fchollet  enw  adam leei enas allos
    return model



def create_extreme_LSTM (LSTM_size, Dense_sizes, embeddings, max_input_length, is_trainable):
    D = embeddings.shape[-1]
    out_dim = 5
    model = Sequential()
    model.add(Embedding(input_dim = embeddings.shape[0],
                        output_dim=D,
                        weights=[embeddings],
                        trainable=is_trainable,
                        input_length = max_input_length))
    model.add(LSTM(LSTM_size))
    model.add(Activation('sigmoid'))
    for Dense_size in Dense_sizes:
        model.add(Dense(Dense_size))
        model.add(Activation('sigmoid'))
    model.add(Dense(out_dim))
    model.add(Activation('linear'))
    model.compile(loss='rmse', optimizer='sgd')
    return model



def create_stacked_LSTM (LSTM_size, Dense_sizes, embeddings, max_input_length, is_trainable):
    D = embeddings.shape[-1]
    out_dim = 5
    model = Sequential()
    model.add(Embedding(input_dim = embeddings.shape[0],
                        output_dim=D,
                        weights=[embeddings],
                        trainable=is_trainable,
                        input_length = max_input_length))
    model.add(LSTM(LSTM_size,activation='sigmoid', return_sequences=True))
    model.add(LSTM(LSTM_size,activation='sigmoid', return_sequences=True))
    model.add(LSTM(LSTM_size,activation='sigmoid', return_sequences=True))
    model.add(LSTM(LSTM_size,activation='sigmoid', return_sequences=False))
    model.add(Activation('sigmoid'))
    for Dense_size in Dense_sizes:
        model.add(Dense(Dense_size))
        model.add(Activation('sigmoid'))
    model.add(Dense(out_dim))
    model.add(Activation('linear'))
    model.compile(loss='rmse', optimizer='sgd')
    return model


