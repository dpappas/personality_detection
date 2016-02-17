#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding=utf-8

__author__ = 'Dimitris'


'''
data = read_vecs('/data_repository/Twitter_data/glove.twitter.27B.200d.txt')
my_voc = create_distinct_words_from_folder( './data/training/', './data/testing/' )
my_emb = get_my_vecs(data,my_voc)
data = None
pickle.dump( data, open( "my_emb.p", "wb" ) )

V, emb = handle_vecs(my_emb)
pickle.dump( V, open( "V.p", "wb" ) )
pickle.dump( emb, open( "emb.p", "wb" ) )

train_indexes = create_indexes_matrix_from_folder(V,'./data/training/')
pickle.dump( train_indexes, open( "train_indexes.p", "wb" ) )

test_indexes = create_indexes_matrix_from_folder(V,'./data/testing/')
pickle.dump( test_indexes, open( "test_indexes.p", "wb" ) )

train = handle_all(scores, train_indexes, max_length)
pickle.dump( train, open( "train.p", "wb" ) )

test = handle_all(scores, test_indexes, max_length)
pickle.dump( test, open( "test.p", "wb" ) )


test = handle_all(scores, test_indexes, max_length)
pickle.dump( test, open( "test.p", "wb" ) )



print('loading my_emb.p')
my_emb = pickle.load( open( "my_emb.p", "rb" ) )



print('loading train_indexes.p')
train_indexes = pickle.load( open( "train_indexes.p", "rb" ) )

print('loading test_indexes.p')
test_indexes = pickle.load( open( "test_indexes.p", "rb" ) )

print('loading scores.p')
scores = pd.read_csv('./data/Personality_scores.csv',index_col= 0,header = 0, delimiter=' ')

max_length = get_max_length(test_indexes,train_indexes)
print(max_length)



'''


import pickle
from aiding_funcs.NN_handling.CNN import create_simple_CNN_2D, create_simple_CNN_2D_no_emb, get_CNN_results
from aiding_funcs.NN_handling.LSTM import create_simple_LSTM, create_extreme_LSTM, create_stacked_LSTM
from aiding_funcs.embeddings_handling import get_the_folds
from aiding_funcs.label_handling import MaxMin

from keras.callbacks import EarlyStopping

print('loading V.p and emb.p')
V = pickle.load( open( "./pickles/V.p", "rb" ) )
emb = pickle.load( open( "./pickles/emb.p", "rb" ) )

print('loading train.p')
train = pickle.load( open( "./pickles/train.p", "rb" ) )

print('loading test.p')
test = pickle.load( open( "./pickles/test.p", "rb" ) )

'''
V = pickle.load( open( "./V.p", "rb" ) )
emb = pickle.load( open( "./emb.p", "rb" ) )
train = pickle.load( open( "./train.p", "rb" ) )
test = pickle.load( open( "./test.p", "rb" ) )

'''


'''
CNN basic
'''

CNN_filters = 10
CNN_rows = 2
Dense_size = 100
max_input_length = test['features'].shape[1]
is_trainable = False
opt = 'adam'
model = create_simple_CNN_2D (CNN_filters, CNN_rows, Dense_size, emb, max_input_length, is_trainable, opt)
t_l, T_l = MaxMin(test['labels'], train['labels'])
get_CNN_results(model,train['features'],T_l,train['features'],T_l)
get_CNN_results(model,train['features'],T_l,test['features'],t_l)

t = model.fit(train['features'],T_l, batch_size=64, nb_epoch=80, validation_split=0.2)
scores = model.evaluate(test['features'],t_l)

'''
CNN basic Trainable
'''

CNN_filters = 10
CNN_rows = 2
Dense_size = 100
max_input_length = test['features'].shape[1]
is_trainable = True
opt = 'adam'
model = create_simple_CNN_2D (CNN_filters, CNN_rows, Dense_size, emb, max_input_length, is_trainable)
get_CNN_results(model,train['features'],train['labels'],test['features'],test['labels'])


'''
CNN basic no embeddings
'''

CNN_filters = 10
CNN_rows = 2
Dense_size = 100
max_input_length = test['features'].shape[1]
in_dim = 1193514
emb_size = 200
opt = 'adam'
model = create_simple_CNN_2D_no_emb (CNN_filters, CNN_rows, Dense_size, max_input_length, emb_size, in_dim)
get_CNN_results(model,train['features'],train['labels'],test['features'],test['labels'])



'''
LSTM basic
'''
is_trainable = False
LSTM_size = 10
Dense_size = 10
max_input_length = test['features'].shape[1]
opt = 'adam'
model = create_simple_LSTM (LSTM_size, Dense_size, emb, max_input_length, is_trainable)
early_stop = EarlyStopping(monitor='val_loss',patience=5)
t = model.fit(
                train['features'],
                train['labels'],
                batch_size=64,
                nb_epoch=50,
                callbacks=[early_stop],
                validation_split=0.2
)
scores = model.evaluate(test['features'], test['labels'])
print(scores)



'''
LSTM basic Trainable
'''
is_trainable = True
LSTM_size = 10
Dense_size = 10
max_input_length = test['features'].shape[1]
opt = 'adam'
model = create_simple_LSTM (LSTM_size, Dense_size, emb, max_input_length, is_trainable)
#early_stop = EarlyStopping(monitor='loss',patience=30)
t = model.fit(
                train['features'],
                train['labels'],
                batch_size=64,
                nb_epoch=200,
                #callbacks=[early_stop],
                validation_split=0.2
)
scores = model.evaluate(test['features'], test['labels'])
results = model.predict(test['features'])
print(scores)






'''
LSTM Insane
'''
is_trainable = False
LSTM_size = 100
max_input_length = test['features'].shape[1]
Dense_sizes = [100,100,100,100,100,100,100,100]
opt = 'adam'
model = create_extreme_LSTM (LSTM_size, Dense_sizes, emb, max_input_length, is_trainable)
early_stop = EarlyStopping(monitor='val_loss',patience=5)
t = model.fit(
                train['features'],
                train['labels'],
                batch_size=64,
                nb_epoch=50,
                callbacks=[early_stop],
                validation_split=0.2
)
scores = model.evaluate(test['features'], test['labels'])
print(scores)



'''
Stacked LSTM
'''

is_trainable = False
LSTM_size = 200
max_input_length = test['features'].shape[1]
Dense_sizes = [100]
model = create_stacked_LSTM (LSTM_size, Dense_sizes, emb, max_input_length, is_trainable)
opt = 'adam'
early_stop = EarlyStopping(monitor='loss',patience=20)
t = model.fit(
                train['features'],
                train['labels'],
                batch_size=64,
                nb_epoch=200,
                callbacks=[early_stop],
                validation_split=0.2
)
scores = model.evaluate(test['features'], test['labels'])
print(scores)


'''

t = len(embeddings[0])

for i in range(len(embeddings)):
    r = embeddings[i]
    if(t!=len(r)):
        print(i)

'''



















































