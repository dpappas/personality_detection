__author__ = 'Dimitris'


import pickle
from aiding_funcs.embeddings_handling import get_the_folds, join_folds
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, KernelCenterer
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import numpy as np
from pprint import pprint

def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def select_features_pca(train_X, test_X, k):
    selector = PCA(n_components=k)
    selector.fit(train_X)
    train_X = selector.transform(train_X)
    test_X = selector.transform(test_X)
    return train_X, test_X

def benchmark(clf, train_X, train_y, test_X, test_y):
    """
    evaluate classification
    """
    clf.fit(train_X, train_y)
    pred = clf.predict(test_X)
    f1 = metrics.f1_score(test_y, pred, average='weighted')
    accuracy = metrics.accuracy_score(test_y, pred)
    print(" Acc: %f "%(accuracy))
    result = {'f1' : f1,'accuracy' : accuracy,'train size' : len(train_y), 'test size' : len(test_y) }
    return result


print('loading test.p')
test = pickle.load( open( "/data/dpappas/Common_Crawl_840B_tokkens_pickles/test.p", "rb" ) )

print('loading train.p')
train = pickle.load( open( "/data/dpappas/Common_Crawl_840B_tokkens_pickles/train.p", "rb" ) )

no_of_folds = 10
folds = get_the_folds(train,no_of_folds)

ret = {}
train_folds = range(9)
train_data = join_folds(folds,train_folds)
validation_data = folds[folds.keys()[-1]]

train_X = train_data['skipthoughts']
train_y = train_data['labels']
test_X = validation_data['skipthoughts']
test_y = validation_data['labels']

ret = {}
for index in range(5):
    train_y_2 = train_y[:,index]
    test_y_2 = test_y[:,index]
    min_err = None
    min_ci = None
    min_gam = None
    for ci in np.arange(0.01,5,0.01):
        for gam in np.arange(0.01,5,0.01):
            svr_rbf = SVR(kernel='rbf', C=ci, gamma=gam)
            rbf_model = svr_rbf.fit(train_X, train_y_2)
            rbf_train = rbf_model.predict(train_X)
            rbf_test = rbf_model.predict(test_X)
            rbf_rmse_t = get_rmse(test_y[:,index], rbf_test)
            if( (min_err == None) or (min_err > rbf_rmse_t) ):
                min_err = rbf_rmse_t
    ret[index] = {
        'ci' : ci,
        'gamma' : gam
    }

pprint(ret)