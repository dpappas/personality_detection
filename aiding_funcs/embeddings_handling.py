#!/usr/bin/env python
#-*- coding: utf-8 -*-
# coding=utf-8

__author__ = 'Dimitris'

import multiprocessing as mp
import re
import numpy as np
import random
from aiding_funcs.file_handling import get_words_from_file
import os

def read_vecs(vecs_file):
    with open(vecs_file, 'r') as f:
        lines = f.readlines()
    data = {}
    for line in lines:
        t = re.split(r'\s+',line.strip())
        data[t[0].decode('utf8')] = [float(x) for x in t[1:]]
    return data

def handle_vecs(data):
    V = list(data.keys())
    embeddings = []
    for l in V:
        embeddings.append(data[l])
    #data = None
    V = np.array(V)
    embeddings = np.array(embeddings)
    return V,embeddings

def create_one_hot_vec(V,words):
    ret = []
    for w in words:
        row = V.shape[0] * [0]
        temp = np.where( V == w )[0]
        if(temp.shape[0]>0):
            index = np.where( V == w )[0][0]
            row[index] = 1
        ret.append(row)
    return np.array(ret)

def get_files_of_folder(dir_path,file_type):
    return [ dir_path+f for f in os.listdir(dir_path) if f.lower().endswith('.'+file_type)]

def create_one_hot_vecs_from_folder(V,dir_path):
    files = get_files_of_folder(dir_path,'txt')
    ret = []
    for f in files:
        w = get_words_from_file(f)
        hot = create_one_hot_vec(V,w)
        ret.append(hot)
        print('finished: '+f)
    return np.array(ret)

def get_index(V,words):
    ret = []
    for w in words:
        temp = np.where( V == w )[0]
        if(temp.shape[0]>0):
            ret.append(temp)
    return np.array(ret)

def create_indexes_matrix_from_folder(V,dir_path):
    files = get_files_of_folder(dir_path,'txt')
    ret = {}
    for f in files:
        w = get_words_from_file(f)
        ind = get_index(V,w)
        key = f.split('/')[-1].replace('.txt','')
        ret[key] = ind
        print('finished: '+f)
    return ret

def create_indexes_matrix_from_file_multithread(V,file,output):
    w = get_words_from_file(file)
    ind = get_index(V,w)
    key = file.split('/')[-1].replace('.txt','')
    output.put((key, ind))
    print('finished: '+file)

def create_indexes_from_folder_multithread(V,dir_path):
    output = mp.Queue()
    files = get_files_of_folder(dir_path,'txt')
    processes = []
    for f in files:
        t = mp.Process(target=create_indexes_matrix_from_file_multithread, args=(V,f, output))
        processes.append(t)
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    results = [output.get() for p in processes]
    ret = {}
    for r in results:
        ret[r[0]] = r[1]
    return ret

def create_distinct_words_from_folder(train_dir_path, test_dir_path):
    ret = []
    files = get_files_of_folder(train_dir_path,'txt')
    files.extend(get_files_of_folder(test_dir_path,'txt'))
    for f in files:
        words = get_words_from_file(f)
        for w in words:
            if( w not in ret):
                ret.append(w)
    return np.array(ret)

def get_my_vecs(data,my_voc):
    ret = {}
    for word in my_voc:
        if(word in data.keys()):
            ret[word] = data[word]
    return ret

def get_max_length(test_indexes,train_indexes):
    mt = max([ test_indexes[k].shape[0] for k in test_indexes.keys()])
    mT = max([ train_indexes[k].shape[0] for k in train_indexes.keys()])
    return max(mt,mT)

def handle_all(scores, indexes, metr):
    row_labels = list(scores.index)
    labels = None
    features = None
    names = None
    for k in indexes.keys():
        i = row_labels.index(k)
        label = np.array(scores.iloc[i])
        feature = indexes[k].T
        temp = np.array((metr - feature.shape[1]) * [0])
        feature = np.append(feature,temp)
        name = k
        if (labels == None):
            labels = label
            features = feature
            names = [name]
        else:
            labels = np.vstack((labels,label))
            features = np.vstack((features,feature))
            names.append(name)
    return {
        'names':names,
        'labels':labels,
        'features':features
    }

def reshape3Dto4D(data_3D):
    return data_3D.reshape((data_3D.shape[0], 1, data_3D.shape[1], data_3D.shape[2]))


'''
Example
t1 = np.array([[1,1,1],[2,2,2],[3,3,3]])
t2 = np.array([[1,1],[2,2],[3,3]])
t3 = np.array([[1],[2],[3]])
t = {'t1':t1,'t2':t2,'t3':t3}
temp = get_the_folds(t,2)

Lambanei ws eisodo ena dictionary me polla numpy arrays kai
epistrefei n folds isokatanemhmena ws
ena neo dictionary me keys 0:n-1 kai se kathe ena apo afta
periexetai ena neo dictionary me keys ta idia me to data alla megethos data.shape[0]/n
'''
def get_the_folds(data,nfolds):
    ndata = {}
    for k in data.keys():
        if(type(data[k]) != np.ndarray):
            data[k] = np.array([data[k]]).T
        ndata[k] = np.copy(data[k])
    for k in ndata.keys():
        ndata[k] = np.array(ndata[k])
    ret = dict()
    m = ndata[ndata.keys()[0]].shape[0]
    for k in ndata.keys():
        if(ndata[k].shape[0] != m):
            print('Error in get_the_folds. \nThe first dim must be the same for all.')
            return None
    for i in range(nfolds):
        ret[i] = None
    i = 0
    temp = ndata[ndata.keys()[0]].shape[0]
    while ( temp>0 ):
        rint = random.randint(0,ndata[ndata.keys()[0]].shape[0]-1)
        if(ret[i] is None):
            ret[i] = {}
            for k in ndata.keys() :
                ret[i][k] = np.array([ndata[k][rint]])
        else:
            for k in ndata.keys() :
                ret[i][k] = np.append(ret[i][k], [ndata[k][rint]] , axis=0)
        for k in ndata.keys() :
            ndata[k] = np.delete(ndata[k], rint, 0)
        temp = ndata[ndata.keys()[0]].shape[0]
        i = (i+1)%nfolds
    return ret

'''

for f in folds.keys():
    for k in folds[f].keys():
        print(folds[f][k].shape)
    print('')

np.vstack( ( np.copy(folds[1]['labels']) , np.copy(folds[8]['labels']) ))

'''

def join_folds(folds, indexes):
    ret = {}
    for k in folds[folds.keys()[0]].keys():
        ret[k] = None
    for i in indexes:
        for k in folds[i].keys():
            if(ret[k] == None):
                ret[k] = np.copy(folds[i][k])
            else:
                ret[k] = np.vstack( (ret[k], np.copy(folds[i][k])) )
    return ret









