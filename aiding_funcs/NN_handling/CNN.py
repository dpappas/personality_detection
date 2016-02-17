#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding=utf-8

__author__ = 'Dimitris'


from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Reshape, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, activity_l1



def create_simple_CNN (nb_filter, filter_length, Dense_size, embeddings, trainable):
    max_features = embeddings.shape[0]
    embedding_dims = embeddings.shape[-1]
    out_dim = 5
    maxlen = 100
    model = Sequential()
    model.add(Embedding(max_features, embedding_dims, weights=[embeddings], input_length = maxlen, trainable=trainable))
    model.add(Convolution1D( nb_filter = nb_filter, filter_length=filter_length, border_mode='valid', subsample_length=1  ))
    sh = model.layers[-1].output_shape
    model.add(MaxPooling1D(pool_length=sh[-2]))
    model.add(Flatten())
    model.add(Dense(Dense_size))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss='rmse', optimizer='sgd')



def create_simple_CNN_2D (CNN_filters, CNN_rows, Dense_size, embeddings, max_input_length, is_trainable):
    D = embeddings.shape[-1]
    cols = D
    out_dim = 5
    model = Sequential()
    model.add(Embedding(input_dim = embeddings.shape[0], output_dim=D, weights=[embeddings], trainable=is_trainable, input_length = max_input_length))
    model.add(Reshape((1, max_input_length, D)))
    model.add(Convolution2D( CNN_filters, CNN_rows, cols, dim_ordering='th', activation='sigmoid' ))
    sh = model.layers[-1].output_shape
    model.add(MaxPooling2D(pool_size=(sh[-2], sh[-1]),dim_ordering = 'th'))
    model.add(Flatten())
    model.add(Dense(Dense_size, activation='sigmoid'))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss='rmse', optimizer='sgd')
    return model


def create_simple_CNN_2D_no_emb (CNN_filters, CNN_rows, Dense_size, max_input_length, emb_size, in_dim):
    out_dim = 5
    model = Sequential()
    model.add(Embedding(input_dim = in_dim, output_dim=emb_size, trainable=True, input_length = max_input_length))
    model.add(Reshape((1, max_input_length, emb_size)))
    model.add(Convolution2D( CNN_filters, CNN_rows, emb_size, dim_ordering='th', activation='sigmoid' ))
    sh = model.layers[-1].output_shape
    model.add(MaxPooling2D(pool_size=(sh[-2], sh[-1]),dim_ordering = 'th'))
    model.add(Flatten())
    model.add(Dense(Dense_size, activation='sigmoid'))
    model.add(Dense(out_dim, activation='linear'))
    model.compile(loss='rmse', optimizer='sgd')
    return model

def get_CNN_results(model,train_x,train_y,test_x,test_y):
    early_stop = EarlyStopping(monitor='loss',patience=20)
    t = model.fit(
                    train_x,
                    train_y,
                    batch_size=64,
                    nb_epoch=80,
                    callbacks=[early_stop],
                    validation_split=0.2
    )
    scores = model.evaluate(test_x, test_y)
    print(scores)





