__author__ = 'Dimitris'

__author__ = 'Dimitris'

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from pprint import pprint


def keras_model():
    from keras.regularizers import l2, activity_l2
    from aiding_funcs.embeddings_handling import get_the_folds, join_folds
    from keras.layers.recurrent import LSTM
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.layers.embeddings import Embedding
    from keras.regularizers import l1, activity_l1
    import pickle
    embeddings = pickle.load( open( "./emb.p", "rb" ) )
    train = pickle.load( open( "./train.p", "rb" ) )
    no_of_folds = 10
    folds = get_the_folds(train,no_of_folds)
    train_data = join_folds(folds,folds.keys()[:-1])
    validation_data = folds[folds.keys()[-1]]
    max_input_length = validation_data['features'].shape[1]
    LSTM_size = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    Dense_size = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    opt = {{choice([ 'adadelta','sgd','rmsprop', 'adagrad', 'adadelta', 'adam'])}}
    is_trainable = {{choice([ True, False ])}}
    D = embeddings.shape[-1]
    out_dim = 5
    model = Sequential()
    model.add(Embedding(input_dim = embeddings.shape[0], output_dim=D, weights=[embeddings], trainable=is_trainable, input_length = max_input_length))
    model.add(LSTM(LSTM_size, activation = 'sigmoid'))
    model.add(Dense(Dense_size, activation = 'sigmoid',W_regularizer=l2({{uniform(0, 1)}}), activity_regularizer=activity_l2({{uniform(0, 1)}})))
    model.add(Dense(out_dim, activation = 'linear',W_regularizer=l2({{uniform(0, 1)}}), activity_regularizer=activity_l2({{uniform(0, 1)}})))
    model.compile(loss='mse', optimizer= opt) # kalutera leei rmsprop o fchollet  enw  adam leei enas allos
    model.fit(train_data['features'], train_data['labels'], nb_epoch=50, show_accuracy=True, verbose=2)
    score = model.evaluate( validation_data['features'], validation_data['labels'])
    #score = model.evaluate( train_data['features'], train_data['labels'])
    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(keras_model, algo=tpe.suggest, max_evals=1000, trials=Trials())
    pprint(best_run)

'''

Best on Validation set

'''

'''

Best on Training set


'''
