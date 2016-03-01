__author__ = 'Dimitris'

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform
from pprint import pprint

def keras_model():
    from keras.models import Sequential, Graph
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
    CNN_filters = {{choice([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100,105,110,115,120,125,130,135,140,145,150,155,160,165,170,175,180,185,190,195,200])}}
    CNN_rows = {{choice([1,2,3,4,5,6,7,8,9,10])}}
    Dense_size = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    Dense_size2 = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    Dense_size3 = {{choice([50, 100, 150, 200, 250, 300, 350, 400, 450, 500])}}
    opt = {{choice([ 'adadelta','sgd', 'adam'])}}
    is_trainable = {{choice([ True, False ])}}
    D = embeddings.shape[-1]
    cols = D
    out_dim = train_data['labels'].shape[-1]
    graph = Graph()
    graph.add_input(name='txt_data', input_shape=[train_data['features'].shape[-1]], dtype='int')
    graph.add_node(Embedding( input_dim = embeddings.shape[0], output_dim=D, weights=[embeddings], trainable=is_trainable, input_length = max_input_length), name='Emb', input='txt_data')
    graph.add_node(Reshape((1, max_input_length, D)), name = "Reshape", input='Emb')
    graph.add_node( Convolution2D(CNN_filters, CNN_rows, cols, activation='sigmoid' ) , name='Conv', input='Reshape')
    sh = graph.nodes['Conv'].output_shape
    graph.add_node(  MaxPooling2D(pool_size=(sh[-2], sh[-1])) ,  name='MaxPool', input='Conv')
    graph.add_node(  Flatten()  ,  name='Flat', input='MaxPool')
    graph.add_node(  Dense(Dense_size, activation='sigmoid',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}}))  ,  name='Dtxt', input='Flat')
    graph.add_node(  Dropout({{uniform(0, 1)}})  ,  name='Dropout1', input='Dtxt')
    graph.add_input(name='av_data', input_shape=[train_data['AV'].shape[-1]])
    graph.add_node(  Dense(Dense_size2, activation='sigmoid',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}}))  ,  name='Dav', input='av_data')
    graph.add_node(  Dropout({{uniform(0, 1)}})  ,  name='Dropout2', input='Dav')
    graph.add_node(  Dense(Dense_size3, activation='sigmoid',W_regularizer=l2({{uniform(0, 1)}}),activity_regularizer=activity_l2({{uniform(0, 1)}})),  name='Dense1', inputs=['Dropout2', 'Dropout1'], merge_mode='concat')
    graph.add_node(  Dropout({{uniform(0, 1)}})  ,  name='Dropout3', input='Dense1')
    graph.add_node(  Dense(out_dim, activation='linear')  ,  name='Dense2', input='Dropout3')
    graph.add_output(name='output', input = 'Dense2')
    graph.compile(optimizer=opt, loss={'output':'rmse'})
    graph.fit(
        {
            'txt_data':train_data['features'],
            'av_data':train_data['AV'],
            'output':train_data['labels']
        },
        nb_epoch=500,
        batch_size=64
    )
    scores = graph.evaluate({'txt_data':validation_data['features'], 'av_data':validation_data['AV'], 'output':validation_data['labels']})

    return {'loss': scores, 'status': STATUS_OK}

if __name__ == '__main__':
    best_run = optim.minimize(keras_model, algo=tpe.suggest, max_evals=1000, trials=Trials())
    pprint(best_run)
