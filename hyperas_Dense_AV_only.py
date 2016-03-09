__author__ = 'Dimitris'

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from pprint import pprint


def keras_model():
    from keras.models import Sequential
    from keras.layers.core import Dense, Reshape, Activation, Flatten, Dropout
    from keras.regularizers import l1, activity_l1, l2, activity_l2
    from aiding_funcs.embeddings_handling import get_the_folds, join_folds
    from aiding_funcs.label_handling import MaxMin, myRMSE, MaxMinFit
    import pickle
    train = pickle.load( open( "/data/dpappas/personality/train.p", "rb" ) )
    no_of_folds = 10
    folds = get_the_folds(train,no_of_folds)
    train_data = join_folds(folds,folds.keys()[:-1])
    validation_data = folds[folds.keys()[-1]]
    mins, maxs = MaxMin(train_data['AV'])
    T_AV =  MaxMinFit(train_data['AV'], mins, maxs)
    Dense_size = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    Dense_size2 = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    Dense_size3 = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    opt = {{choice([ 'adadelta','sgd','rmsprop', 'adagrad', 'adadelta', 'adam'])}}
    out_dim = 5
    model = Sequential()
    model.add(Dense(Dense_size, activation='sigmoid',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}}),input_dim = train_data['AV'].shape[-1] ))
    model.add(Dense(Dense_size2, activation='sigmoid',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}})))
    model.add(Dense(Dense_size3, activation='sigmoid',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}})))
    model.add(Dense(out_dim, activation='linear',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}})))
    model.compile(loss='rmse', optimizer=opt)
    model.fit(T_AV, train_data['labels'], nb_epoch=500, show_accuracy=False, verbose=2)
    #score = model.evaluate( validation_data['features'], validation_data['labels'])
    score = model.evaluate( T_AV, train_data['labels'])
    print("score : " +str(score))
    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(keras_model, algo=tpe.suggest, max_evals=1000, trials=Trials())
    pprint(best_run)
