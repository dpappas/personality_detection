__author__ = 'Dimitris'

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from pprint import pprint


def keras_model():
    from keras.models import Sequential
    from keras.layers.core import Dense
    from keras.regularizers import l2, activity_l2
    from aiding_funcs.embeddings_handling import get_the_folds, join_folds
    from aiding_funcs.label_handling import MaxMin, MaxMinFit
    import pickle
    print('loading test.p')
    test = pickle.load( open( "/data/dpappas/Common_Crawl_840B_tokkens_pickles/test.p", "rb" ) )
    print('loading train.p')
    train = pickle.load( open( "/data/dpappas/Common_Crawl_840B_tokkens_pickles/train.p", "rb" ) )
    no_of_folds = 10
    folds = get_the_folds(train,no_of_folds)
    train_data = join_folds(folds,folds.keys()[:-1])
    validation_data = folds[folds.keys()[-1]]
    mins, maxs = MaxMin(train_data['labels'])
    T_l = MaxMinFit(train_data['labels'], mins, maxs)
    t_l = MaxMinFit(validation_data['labels'], mins, maxs)


    Dense_size = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    Dense_size2 = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    opt = {{choice([ 'adadelta','sgd','rmsprop', 'adagrad', 'adadelta', 'adam'])}}
    out_dim = 5
    activity_l2_0 = {{uniform(0, 1)}}
    activity_l2_1 = {{uniform(0, 1)}}
    activity_l2_2 = {{uniform(0, 1)}}
    l2_0 = {{uniform(0, 1)}}
    l2_1 = {{uniform(0, 1)}}
    l2_2 = {{uniform(0, 1)}}

    model = Sequential()
    model.add(Dense(Dense_size, activation='sigmoid',W_regularizer=l2(l2_0),activity_regularizer=activity_l2(activity_l2_0),input_dim = train_data['skipthoughts'].shape[-1] ))
    model.add(Dense(Dense_size2, activation='sigmoid',W_regularizer=l2(l2_1),activity_regularizer=activity_l2(activity_l2_1)))
    model.add(Dense(out_dim, activation='linear',W_regularizer=l2(l2_2),activity_regularizer=activity_l2(activity_l2_2)))
    model.compile(loss='rmse', optimizer=opt)

    #model.fit(train_data['skipthoughts'], train_data['labels'], nb_epoch=500, show_accuracy=False, verbose=2)
    #score = model.evaluate( train_data['skipthoughts'], train_data['labels'])

    model.fit(train_data['skipthoughts'], T_l, nb_epoch=500, show_accuracy=False, verbose=2)
    score = model.evaluate( train_data['skipthoughts'], T_l)

    print("score : " +str(score))
    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(keras_model, algo=tpe.suggest, max_evals=2000, trials=Trials())
    pprint(best_run)



'''
{'Dense_size': 3,               200
 'Dense_size2': 5,              300
 'activity_l2_0': 0.05188918775936191,
 'activity_l2_1': 0.45047635433513034,
 'activity_l2_2': 0.0005117368813977515,
 'l2_0': 0.8718331552337388,
 'l2_1': 0.5807575417209597,
 'l2_2': 0.48965647861094225,
 'opt': 5}                      'adam'

'''