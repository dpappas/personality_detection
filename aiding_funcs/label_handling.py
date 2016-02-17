__author__ = 'Dimitris'

import numpy as np

def get_maxs(test_labels,train_labels):
    maxs_t = [np.max(test_labels[:,i]) for i in range(test_labels.shape[1])]
    maxs_T = [np.max(train_labels[:,i]) for i in range(train_labels.shape[1])]
    maxs = [ max( maxs_t[i], maxs_T[i] ) for i in range(len(maxs_t))]
    return maxs

def get_mins(test_labels,train_labels):
    mins_t = [np.min(test_labels[:,i]) for i in range(test_labels.shape[1])]
    mins_T = [np.min(train_labels[:,i]) for i in range(train_labels.shape[1])]
    mins = [ max( mins_t[i], mins_T[i] ) for i in range(len(mins_t))]
    return mins

'''

test_labels = test['labels']
train_labels = train['labels']
MaxMin(test_labels,train_labels)
'''

def MaxMin(test_labels,train_labels):
   maxs = get_maxs(test_labels,train_labels)
   mins = get_mins(test_labels,train_labels)
   t = np.array(test_labels)
   for i in range(t.shape[1]):
       for i in range(t.shape[1]):
        t[:,i] = (t[:,i] - mins[i])/(maxs[i]-mins[i])
   T = np.array(train_labels)
   for i in range(T.shape[1]):
       T[:,i] = (T[:,i] - mins[i])/(maxs[i]-mins[i])
   return t,T

