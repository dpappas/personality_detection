__author__ = 'Dimitris'

import pickle
from aiding_funcs.embeddings_handling import get_the_folds, join_folds
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer, KernelCenterer
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import numpy as np

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
for i in range(max(folds.keys())+1):
    train_folds = range(i+1)
    train_data = join_folds(folds,train_folds)
    train_X = train_data['skipthoughts']
    train_y = train_data['labels']
    test_X = test['skipthoughts']
    test_y = test['labels']
    for index in range(5):
        train_y_2 = train_y[:,index]
        test_y_2 = test_y[:,index]
        #normalizer = MinMaxScaler(feature_range=(0.0,1.0))
        #train_y_2 = normalizer.fit_transform(train_y_2.reshape(-1, 1))
        #test_y_2 = normalizer.transform(test_y_2.reshape(-1, 1))
        #k = 0.2
        #train_X, test_X = select_features_pca(train_X, test_X, k)
        svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
        svr_lin = SVR(kernel='linear', C=1e3)
        svr_poly = SVR(kernel='poly', C=1e3, degree=2)
        #
        rbf_model = svr_rbf.fit(train_X, train_y_2)
        rbf_train = rbf_model.predict(train_X)
        rbf_test = rbf_model.predict(test_X)
        #rbf_train = normalizer.inverse_transform(rbf_train.reshape(-1, 1))
        #rbf_test = normalizer.inverse_transform(rbf_test.reshape(-1, 1))
        rbf_rmse_T = get_rmse(train_y[:,index], rbf_train)
        rbf_rmse_t = get_rmse(test_y[:,index], rbf_test)
        #
        lin_model = svr_lin.fit(train_X, train_y_2)
        lin_train = lin_model.predict(train_X)
        lin_test = lin_model.predict(test_X)
        #lin_train = normalizer.inverse_transform(lin_train.reshape(-1, 1))
        #lin_test = normalizer.inverse_transform(lin_test.reshape(-1, 1))
        lin_rmse_T = get_rmse(train_y[:,index], lin_train)
        lin_rmse_t = get_rmse(test_y[:,index], lin_test)
        #
        poly_model = svr_poly.fit(train_X, train_y_2)
        poly_train = poly_model.predict(train_X)
        poly_test = poly_model.predict(test_X)
        #poly_train = normalizer.inverse_transform(poly_train.reshape(-1, 1))
        #poly_test = normalizer.inverse_transform(poly_test.reshape(-1, 1))
        poly_rmse_T = get_rmse(train_y[:,index], poly_train)
        poly_rmse_t = get_rmse(test_y[:,index], poly_test)
        #
        print("Index : "+str(index)+"| Fold "+str(i)+" | RBF | test : "+str(rbf_rmse_t)+" | train : "+str(rbf_rmse_T))
        print("Index : "+str(index)+"| Fold "+str(i)+" | LINEAR | test : "+str(lin_rmse_t)+" | train : "+str(lin_rmse_T))
        print("Index : "+str(index)+"| Fold "+str(i)+" | POLYNOMIAL | test : "+str(poly_rmse_t)+" | train : "+str(poly_rmse_T))



