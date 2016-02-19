#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding=utf-8

__author__ = 'Dimitris'


import pickle
from aiding_funcs.NN_handling.CNN import create_simple_CNN_2D, create_CNN
from aiding_funcs.label_handling import MaxMin, myRMSE, MaxMinFit
from aiding_funcs.embeddings_handling import get_the_folds, join_folds

print('loading V.p and emb.p')
V = pickle.load( open( "./V.p", "rb" ) )
emb = pickle.load( open( "./emb.p", "rb" ) )

print('loading train.p')
train = pickle.load( open( "./train.p", "rb" ) )

print('loading test.p')
test = pickle.load( open( "./test.p", "rb" ) )


no_of_folds = 10
folds = get_the_folds(train,no_of_folds)

# train_data = join_folds(folds,[0,1])
# test me afto

train_data = join_folds(folds,folds.keys()[:-1])

validation_data = folds[folds.keys()[-1]]




'''

V = pickle.load( open( "./pickles/V.p", "rb" ) )
emb = pickle.load( open( "./pickles/emb.p", "rb" ) )
train = pickle.load( open( "./pickles/train.p", "rb" ) )
test = pickle.load( open( "./pickles/test.p", "rb" ) )

'''


'''
CNN
'''

mins, maxs = MaxMin(train_data['labels'])
T_l = MaxMinFit(train_data['labels'], mins, maxs)
t_l = MaxMinFit(validation_data['labels'], mins, maxs)


Dense_sizes = [100]
Dense_l2_regularizers = [0.01,0.01]
Dense_acivity_l2_regularizers = [0.01,0.01]
CNN_filters = 10
CNN_rows = 2
Dense_size = 100
max_input_length = test['features'].shape[1]
is_trainable = False
opt = 'adadelta' #sgd, rmsprop, adagrad, adadelta, adam

model = create_CNN( CNN_filters,  CNN_rows, Dense_sizes, Dense_l2_regularizers,  Dense_acivity_l2_regularizers, emb,  max_input_length, is_trainable, opt)
t = model.fit( train_data['features'], T_l, batch_size=64, nb_epoch=200 ,validation_data=(validation_data['features'],t_l))
#t = model.fit( train_data['features'], T_l, batch_size=64, nb_epoch=1000 , validation_split=0.2)

pred = model.predict(validation_data['features'])
print(myRMSE(pred, t_l, mins, maxs))
pred = model.predict(train_data['features'])
print(myRMSE(pred, T_l, mins, maxs))

scores = model.evaluate(train_data['features'],T_l)
print(scores)
scores = model.evaluate(validation_data['features'],t_l)
print(scores)




'''

'''

mins, maxs = MaxMin(train['labels'])
T_l = MaxMinFit(train['labels'], mins, maxs)
t_l = MaxMinFit(test['labels'], mins, maxs)

Dense_sizes = [100,100,100]
Dense_l2_regularizers = [0.01,0.01,0.01,0.01]
Dense_acivity_l2_regularizers = [0.01,0.01,0.01,0.01]
CNN_filters = 10
CNN_rows = 2
max_input_length = test['features'].shape[1]
is_trainable = False
opt = 'adadelta' #sgd, rmsprop, adagrad, adadelta, adam

model = create_CNN( CNN_filters, CNN_rows, Dense_sizes, Dense_l2_regularizers, Dense_acivity_l2_regularizers, emb, max_input_length, is_trainable,opt)
t = model.fit( train['features'], T_l, batch_size=64, nb_epoch=1000)
scores = model.evaluate(train['features'],T_l)
print(scores)
scores = model.evaluate(test['features'],t_l)
print(scores)



'''
Results on Folds with embeddings
'''
ret = []
weights = None
for i in range(max(folds.keys())):
    train_folds = range(i+1)
    train_data = join_folds(folds,train_folds)
    mins, maxs = MaxMin(train_data['labels'])
    T_l = MaxMinFit(train_data['labels'], mins, maxs)
    t_l = MaxMinFit(test['labels'], mins, maxs)
    Dense_sizes = [100,100,100]
    Dense_l2_regularizers = [0.01,0.01,0.01,0.01]
    Dense_acivity_l2_regularizers = [0.01,0.01,0.01,0.01]
    CNN_filters = 10
    CNN_rows = 2
    max_input_length = test['features'].shape[1]
    is_trainable = False
    opt = 'adadelta' #sgd, rmsprop, adagrad, adadelta, adam
    model = create_CNN( CNN_filters, CNN_rows, Dense_sizes, Dense_l2_regularizers, Dense_acivity_l2_regularizers, emb, max_input_length, is_trainable,opt)
    if(weights!=None):
        model.set_weights(weights)
    else:
        weights = model.get_weights()
    t = model.fit( train_data['features'], T_l, batch_size=64, nb_epoch=200)
    scores_on_train = model.evaluate(train_data['features'],T_l)
    scores_on_test = model.evaluate(test['features'],t_l)
    print('mse on train : ' + str(scores_on_train))
    print('mse on test : ' + str(scores_on_train))
    print('')
    ret.append([scores_on_train, scores_on_test])



'''
No Embeddings
'''

Dense_sizes = [100]
Dense_l2_regularizers = [0.01,0.01]
Dense_acivity_l2_regularizers = [0.01,0.01]
CNN_filters = 10
CNN_rows = 2
max_input_length = test['features'].shape[1]
is_trainable = False
opt = 'adadelta' #sgd, rmsprop, adagrad, adadelta, adam
model = create_CNN( CNN_filters,  CNN_rows, Dense_sizes, Dense_l2_regularizers,  Dense_acivity_l2_regularizers, embeddings=None ,  max_input_length= max_input_length, is_trainable=is_trainable, opt=opt, emb_size=200)
t = model.fit( train_data['features'], T_l, batch_size=64, nb_epoch=200 ,validation_data=(validation_data['features'],t_l))
#t = model.fit( train_data['features'], T_l, batch_size=64, nb_epoch=1000 , validation_split=0.2)

pred = model.predict(validation_data['features'])
print(myRMSE(pred, t_l, mins, maxs))
pred = model.predict(train_data['features'])
print(myRMSE(pred, T_l, mins, maxs))

scores = model.evaluate(train_data['features'],T_l)
print(scores)
scores = model.evaluate(validation_data['features'],t_l)
print(scores)



'''
Results on Folds without embeddings
'''

ret = []
weights = None
for i in range(max(folds.keys())):
    train_folds = range(i+1)
    train_data = join_folds(folds,train_folds)
    mins, maxs = MaxMin(train_data['labels'])
    T_l = MaxMinFit(train_data['labels'], mins, maxs)
    t_l = MaxMinFit(test['labels'], mins, maxs)
    Dense_sizes = [100,100,100]
    Dense_l2_regularizers = [0.01,0.01,0.01,0.01]
    Dense_acivity_l2_regularizers = [0.01,0.01,0.01,0.01]
    CNN_filters = 10
    CNN_rows = 2
    max_input_length = test['features'].shape[1]
    is_trainable = False
    opt = 'adadelta' #sgd, rmsprop, adagrad, adadelta, adam
    emb_size = 200
    model = create_CNN( CNN_filters, CNN_rows, Dense_sizes, Dense_l2_regularizers, Dense_acivity_l2_regularizers, None, max_input_length, is_trainable,opt, emb_size)
    if(weights!=None):
        model.set_weights(weights)
    else:
        weights = model.get_weights()
    t = model.fit( train_data['features'], T_l, batch_size=64, nb_epoch=200)
    scores_on_train = model.evaluate(train_data['features'],T_l)
    scores_on_test = model.evaluate(test['features'],t_l)
    print('mse on train : ' + str(scores_on_train))
    print('mse on test : ' + str(scores_on_train))
    print('')
    ret.append([scores_on_train, scores_on_test])

