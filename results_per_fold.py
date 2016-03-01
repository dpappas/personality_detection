__author__ = 'Dimitris'

import gc
gc.enable()

from aiding_funcs.NN_handling.CNN import create_CNN
from aiding_funcs.label_handling import MaxMin, myRMSE, MaxMinFit

from aiding_funcs.embeddings_handling import get_the_folds, join_folds
import pickle
from pprint import pprint

print('loading V.p and emb.p')
V = pickle.load( open( "/data/dpappas/personality/V.p", "rb" ) )
emb = pickle.load( open( "/data/dpappas/personality/emb.p", "rb" ) )

print('loading train.p')
train = pickle.load( open( "/data/dpappas/personality/train.p", "rb" ) )

print('loading test.p')
test = pickle.load( open( "/data/dpappas/personality/test.p", "rb" ) )

no_of_folds = 10
folds = get_the_folds(train,no_of_folds)

ret = []
weights = None
for i in range(max(folds.keys())):
    train_folds = range(i+1)
    train_data = join_folds(folds,train_folds)
    mins, maxs = MaxMin(train_data['labels'])
    T_l = MaxMinFit(train_data['labels'], mins, maxs)
    t_l = MaxMinFit(test['labels'], mins, maxs)
    Dense_sizes = [300]
    Dense_l2_regularizers = [0.37173327555716984,0.000165584846072854]
    Dense_acivity_l2_regularizers = [0.9593094177755246,0.0011426757779919388]
    CNN_filters = 5
    CNN_rows = 6
    max_input_length = test['features'].shape[1]
    is_trainable = True
    opt = 'adadelta' #sgd, rmsprop, adagrad, adadelta, adam
    model = create_CNN( CNN_filters, CNN_rows, Dense_sizes, Dense_l2_regularizers, Dense_acivity_l2_regularizers, emb, max_input_length, is_trainable,opt)
    if(weights!=None):
        model.set_weights(weights)
    else:
        weights = model.get_weights()
    t = model.fit( train_data['features'], T_l, batch_size=64, nb_epoch=1200)
    scores_on_train = model.evaluate(train_data['features'],T_l)
    scores_on_test = model.evaluate(test['features'],t_l)
    print('mse on train : ' + str(scores_on_train))
    print('mse on test : ' + str(scores_on_test))
    print('')
    model = None
    gc.collect()
    ret.append([scores_on_train, scores_on_test])

pprint(ret)