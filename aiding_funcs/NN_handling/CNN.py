#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding=utf-8

__author__ = 'Dimitris'


from keras.models import Sequential, Graph
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Reshape, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import EarlyStopping
from keras.regularizers import l1, activity_l1, l2, activity_l2


def create_CNN(
        CNN_filters,                        # # of filters
        CNN_rows,                           # # of rows per filter
        Dense_sizes,                        # matrix of intermediate Dense layers
        Dense_l2_regularizers,              # matrix with the l2 regularizers for the dense layers
        Dense_acivity_l2_regularizers,      # matrix with the l2 activity regularizers for the dense layers
        embeddings,                         # pretrained embeddings or None if there are not any
        max_input_length,                   # maximum length of sentences
        is_trainable,                       # True if the embedding layer is trainable
        opt = 'sgd',                        # optimizer
        emb_size = 200,                      # embedding size if embeddings not given
        input_dim = 500                      # input dimention if embeddings not given
    ):
    out_dim = 5
    model = Sequential()
    if(embeddings != None):
        D = embeddings.shape[-1]
        cols = D
        model.add(Embedding( input_dim = embeddings.shape[0], output_dim=D, weights=[embeddings], trainable=is_trainable, input_length = max_input_length))
    else:
        D = emb_size
        cols = D
        model.add( Embedding( input_dim = input_dim, output_dim=D, trainable=True, input_length = max_input_length ) )
    model.add(Reshape((1, max_input_length, D)))
    model.add(Convolution2D( CNN_filters, CNN_rows, cols, dim_ordering='th', activation='sigmoid' ))
    sh = model.layers[-1].output_shape
    model.add(MaxPooling2D(pool_size=(sh[-2], sh[-1]),dim_ordering = 'th'))
    model.add(Flatten())
    for i in range(len(Dense_sizes)):
        Dense_size = Dense_sizes[i]
        l2r = Dense_l2_regularizers[i]
        l2ar = Dense_acivity_l2_regularizers[i]
        model.add(
            Dense(
                Dense_size,
                activation = 'sigmoid',
                W_regularizer=l2(l2r),
                activity_regularizer=activity_l2(l2ar)
            )
        )
    l2r = Dense_l2_regularizers[-1]
    l2ar = Dense_acivity_l2_regularizers[-1]
    model.add(
        Dense(
            out_dim,
            activation='linear',
                W_regularizer=l2(l2r),
                activity_regularizer=activity_l2(l2ar)
        )
    )
    model.compile(loss='mse', optimizer=opt)
    return model

def create_simple_CNN (nb_filter, filter_length, Dense_size, embeddings, trainable, opt = 'sgd'):
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
    model.compile(loss='mse', optimizer=opt)



def create_simple_CNN_2D (CNN_filters, CNN_rows, Dense_size, embeddings, max_input_length, is_trainable, opt = 'sgd'):
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
    model.compile(loss='mse', optimizer=opt)
    return model


def create_simple_CNN_2D_no_emb (CNN_filters, CNN_rows, Dense_size, max_input_length, emb_size, in_dim, opt = 'sgd'):
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
    model.compile(loss='mse', optimizer=opt)
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


def CNN_multimodal_model( T_labels, T_text, T_av, Dense_size, CNN_rows, filters, emb, is_trainable=False, max_input_length = 100):
    out_dim = T_labels.shape[-1]
    D = emb.shape[-1]
    cols = D
    graph = Graph()
    graph.add_input(name='txt_data', input_shape=[T_text.shape[-1]], dtype='int')
    graph.add_node(Embedding( input_dim = emb.shape[0], output_dim=D, weights=[emb], trainable=is_trainable, input_length = max_input_length), name='Emb', input='txt_data')
    graph.add_node(Reshape((1, max_input_length, D)), name = "Reshape", input='Emb')
    graph.add_node( Convolution2D(filters, CNN_rows, cols, activation='sigmoid' ) , name='Conv', input='Reshape')
    sh = graph.nodes['Conv'].output_shape
    graph.add_node(  MaxPooling2D(pool_size=(sh[-2], sh[-1])) ,  name='MaxPool', input='Conv')
    graph.add_node(  Flatten()  ,  name='Flat', input='MaxPool')
    graph.add_node(  Dense(300, activation='sigmoid')  ,  name='Dtxt', input='Flat')
    graph.add_input(name='av_data', input_shape=[T_av.shape[-1]])
    graph.add_node(  Dense(300, activation='sigmoid')  ,  name='Dav', input='av_data')
    graph.add_node(  Dense(Dense_size, activation='sigmoid'),  name='Dense1', inputs=['Dav', 'Dtxt'], merge_mode='concat')
    graph.add_node(  Dropout(0.5)  ,  name='Dropout', input='Dense1')
    graph.add_node(  Dense(out_dim, activation='linear')  ,  name='Dense2', input='Dropout')
    graph.add_output(name='output', input = 'Dense2')
    graph.compile(optimizer='adadelta', loss={'output':'rmse'})
    return graph



