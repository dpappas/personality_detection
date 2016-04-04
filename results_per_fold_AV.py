__author__ = 'Dimitris'



'''
Results on Folds AV
'''

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2, activity_l2
from aiding_funcs.embeddings_handling import get_the_folds, join_folds
import pickle
from aiding_funcs.label_handling import MaxMin, MaxMinFit

print('loading test.p')
test = pickle.load( open( "/data/dpappas/personality/test.p", "rb" ) )

print('loading train.p')
train = pickle.load( open( "/data/dpappas/personality/train.p", "rb" ) )

no_of_folds = 10
folds = get_the_folds(train,no_of_folds)
mins, maxs = MaxMin(train['AV'])
Dense_size = 100
Dense_size2 = 500
Dense_size3 = 300
opt = 'adadelta'
out_dim = 5
activity_l2_0 = 0.0014082120886226845
activity_l2_1 = 0.16904326507336032
activity_l2_2 = 0.6023544163244046
activity_l2_3 = 0.00012911700632491275
l2_0 = 0.6709001693961152
l2_1 = 0.3174644779626695
l2_2 = 0.13100661950020417
l2_3 = 0.3016331305514661

ret = []
weights = None
for i in range(max(folds.keys())+1):
    train_folds = range(i+1)
    train_data = join_folds(folds,train_folds)
    T_AV =  MaxMinFit(train_data['AV'], mins, maxs)
    t_AV =  MaxMinFit(test['AV'], mins, maxs)
    model = Sequential()
    model.add(Dense(Dense_size, activation='sigmoid',W_regularizer=l2(l2_0),activity_regularizer=activity_l2(activity_l2_0),input_dim = train_data['AV'].shape[-1] ))
    model.add(Dense(Dense_size2, activation='sigmoid',W_regularizer=l2(l2_1),activity_regularizer=activity_l2(activity_l2_1)))
    model.add(Dense(Dense_size3, activation='sigmoid',W_regularizer=l2(l2_2),activity_regularizer=activity_l2(activity_l2_2)))
    model.add(Dense(out_dim, activation='linear',W_regularizer=l2(l2_3),activity_regularizer=activity_l2(activity_l2_3)))
    model.compile(loss='rmse', optimizer=opt)
    if(weights!=None):
        model.set_weights(weights)
    else:
        weights = model.get_weights()
    model.fit(T_AV, train_data['labels'], nb_epoch=2500, show_accuracy=False, verbose=2, batch_size=train_data['labels'].shape[0]) # default batch_size=128
    score = model.evaluate( T_AV, train_data['labels'])
    score2 = model.evaluate( t_AV, test['labels'])
    print("score on train : " +str(score))
    print("score on test : " +str(score2))
    ret.append([score, score2])


