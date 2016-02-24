__author__ = 'Dimitris'

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from pprint import pprint


def keras_model():
    from keras.models import Sequential
    from keras.layers.embeddings import Embedding
    from keras.layers.convolutional import Convolution2D, MaxPooling2D
    from keras.layers.core import Dense, Reshape, Activation, Flatten, Dropout
    from keras.regularizers import l1, activity_l1, l2, activity_l2
    from aiding_funcs.embeddings_handling import get_the_folds, join_folds
    import pickle
    embeddings = pickle.load( open( "/data/dpappas/personality/emb.p", "rb" ) )
    train = pickle.load( open( "/data/dpappas/personality/train.p", "rb" ) )
    no_of_folds = 10
    folds = get_the_folds(train,no_of_folds)
    train_data = join_folds(folds,folds.keys()[:-1])
    validation_data = folds[folds.keys()[-1]]
    max_input_length = validation_data['features'].shape[1]
    CNN_filters = {{choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95])}}
    CNN_rows = {{choice([1,2,3,4,5,6])}}
    Dense_size = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    opt = {{choice([ 'adadelta','sgd','rmsprop', 'adagrad', 'adadelta', 'adam'])}}
    is_trainable = {{choice([ True, False ])}}
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
    model.add(Dense(Dense_size, activation='sigmoid',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}})))
    model.add(Dense(out_dim, activation='linear',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}})))
    model.compile(loss='mse', optimizer=opt)
    model.fit(train_data['features'], train_data['labels'], nb_epoch=50, show_accuracy=False, verbose=2)
    #score = model.evaluate( validation_data['features'], validation_data['labels'])
    score = model.evaluate( train_data['features'], train_data['labels'])
    return {'loss': score, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(keras_model, algo=tpe.suggest, max_evals=1000, trials=Trials())
    pprint(best_run)

'''

Best on Validation set

{'CNN_filters': 17,                         aka 90
 'CNN_rows': 5,                             aka 6
 'Dense_size': 9,                           500
 'activity_l2': 0.569606301223449,
 'is_trainable': 0,                         TRUE
 'l2': 0.36456791622174234,
 'opt': 2}                                  aka 'rmsprop'


'''

'''

Best on Training set

{'CNN_filters': 0,                                      5
 'CNN_rows': 5,                                         6
 'Dense_size': 5,                                       300
 'activity_l2': 0.9593094177755246,
 'activity_l2_1': 0.0011426757779919388,
 'is_trainable': 1,                                     False
 'l2': 0.37173327555716984,
 'l2_1': 0.000165584846072854,
 'opt': 0                                               adadelta
}

'''
