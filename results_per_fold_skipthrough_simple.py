__author__ = 'Dimitris'


from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.regularizers import l2, activity_l2
from aiding_funcs.embeddings_handling import get_the_folds, join_folds
import pickle
from aiding_funcs.label_handling import MaxMin, MaxMinFit


print('loading test.p')
test = pickle.load( open( "/data/dpappas/Common_Crawl_840B_tokkens_pickles/test.p", "rb" ) )

print('loading train.p')
train = pickle.load( open( "/data/dpappas/Common_Crawl_840B_tokkens_pickles/train.p", "rb" ) )

no_of_folds = 10
folds = get_the_folds(train,no_of_folds)
Dense_size = 5
Dense_size2 = 5
opt = 'adadelta'
out_dim = 5
activity_l2_0 = 0.0001
activity_l2_1 = 0.0001
activity_l2_2 = 0.0001
l2_0 = 0.0001
l2_1 = 0.0001
l2_2 = 0.0001


ret = []
weights = None
for i in range(max(folds.keys())+1):
    train_folds = range(i+1)
    train_data = join_folds(folds,train_folds)
    model = Sequential()
    model.add(Dense(Dense_size, activation='sigmoid',W_regularizer=l2(l2_0),activity_regularizer=activity_l2(activity_l2_0),input_dim = train_data['skipthoughts'].shape[-1] ))
    model.add(Dense(Dense_size2, activation='sigmoid',W_regularizer=l2(l2_1),activity_regularizer=activity_l2(activity_l2_1)))
    model.add(Dense(out_dim, activation='linear',W_regularizer=l2(l2_2),activity_regularizer=activity_l2(activity_l2_2)))
    model.compile(loss='rmse', optimizer=opt)
    if(weights!=None):
        model.set_weights(weights)
    else:
        weights = model.get_weights()
    model.fit(train_data['skipthoughts'], train_data['labels'], nb_epoch=10000, show_accuracy=False, verbose=2, batch_size=train_data['labels'].shape[0]) # default batch_size=128
    #model.fit(train_data['skipthoughts'], train_data['labels'], nb_epoch=10000, show_accuracy=False, verbose=2, batch_size = 32) # default batch_size=128
    score = model.evaluate( train_data['skipthoughts'], train_data['labels'])
    score2 = model.evaluate( test['skipthoughts'], test['labels'])
    print("score on train : " +str(score))
    print("score on test : " +str(score2))
    ret.append([score, score2])





