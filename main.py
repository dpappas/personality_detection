#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding=utf-8

__author__ = 'Dimitris'


import pickle
from aiding_funcs.NN_handling.CNN import create_simple_CNN_2D, create_CNN
from aiding_funcs.label_handling import MaxMin, myRMSE, MaxMinFit

print('loading V.p and emb.p')
V = pickle.load( open( "./V.p", "rb" ) )
emb = pickle.load( open( "./emb.p", "rb" ) )

print('loading train.p')
train = pickle.load( open( "./train.p", "rb" ) )

print('loading test.p')
test = pickle.load( open( "./test.p", "rb" ) )

'''

V = pickle.load( open( "./pickles/V.p", "rb" ) )
emb = pickle.load( open( "./pickles/emb.p", "rb" ) )
train = pickle.load( open( "./pickles/train.p", "rb" ) )
test = pickle.load( open( "./pickles/test.p", "rb" ) )

'''


'''
CNN
'''

mins, maxs = MaxMin(train['labels'])
T_l = MaxMinFit(train['labels'], mins, maxs)
t_l = MaxMinFit(test['labels'], mins, maxs)


Dense_sizes = [100,100,100]
Dense_l2_regularizers = [0.5,0.5,0.5,0.5]
Dense_acivity_l2_regularizers = [0.5,0.5,0.5,0.5]
CNN_filters = 10
CNN_rows = 2
Dense_size = 100
max_input_length = test['features'].shape[1]
is_trainable = False
opt = 'adam'
model = create_CNN(
        CNN_filters,                        # # of filters
        CNN_rows,                           # # of rows per filter
        Dense_sizes,                        # matrix of intermediate Dense layers
        Dense_l2_regularizers,              # matrix with the l2 regularizers for the dense layers
        Dense_acivity_l2_regularizers,      # matrix with the l2 activity regularizers for the dense layers
        emb,                         # pretrained embeddings or None if there are not any
        max_input_length,                   # maximum length of sentences
        is_trainable,                       # True if the embedding layer is trainable
        opt = 'sgd',                        # optimizer
    )
t = model.fit(
    train['features'],
    T_l,
    batch_size=64,
    nb_epoch=1000,
    validation_split=0.2
)
pred = model.predict(test['features'])
print(myRMSE(pred, t_l, mins, maxs))
pred = model.predict(train['features'])
print(myRMSE(pred, T_l, mins, maxs))

scores = model.evaluate(train['features'],T_l)








