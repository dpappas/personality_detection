__author__ = 'Dimitris'

import numpy as np

def get_maxs(train_labels):
    maxs_T = [np.max(train_labels[:,i]) for i in range(train_labels.shape[1])]
    return maxs_T

def get_mins(train_labels):
    mins_T = [np.min(train_labels[:,i]) for i in range(train_labels.shape[1])]
    return mins_T



'''

test_labels = test['labels']
train_labels = train['labels']
MaxMin(test_labels,train_labels)

'''

def MaxMin(train_labels):
    maxs = get_maxs(train_labels)
    mins = get_mins(train_labels)
    return mins, maxs

def MaxMinFit(mat, mins, maxs):
    t = np.array(mat)
    for i in range(t.shape[1]):
       t[:,i] = (t[:, i] - mins[i])/(maxs[i]-mins[i])
    return t

def MaxMinAll(test_labels,train_labels):
   mins, maxs = MaxMin(train_labels)
   t = MaxMinFit(test_labels, mins, maxs)
   T = MaxMinFit(train_labels, mins, maxs)
   return t,T

def MaxMinReverse(mat,mins,maxs):
    t = np.array(mat)
    for i in range(t.shape[1]):
        t[:,i] = (t[:, i] * (maxs[i]-mins[i]))+mins[i]
    return t

def RMSE(prd,true):
    rmse = np.array(prd.shape[1]*[0.0])
    for i in range(prd.shape[1]):
        temp = (prd[:,i] - true[:,i])**2
        temp = np.sum(temp) / temp.shape[0]
        temp = temp**(1.0/2.0)
        rmse[i] = temp
    return rmse

def myRMSE(pred, true, mins, maxs):
    prd = MaxMinReverse(pred,mins,maxs)
    tr = MaxMinReverse(true,mins,maxs)
    return RMSE(prd,tr)

