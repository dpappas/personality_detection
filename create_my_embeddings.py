__author__ = 'Dimitris'


#ls /data_repository/data/
#glove.42B.300d.txt  glove.6B.300d.txt  glove.840B.300d.txt

import pickle
import pandas as pd

from aiding_funcs.embeddings_handling import read_vecs, create_distinct_words_from_folder, handle_vecs, get_my_vecs
from aiding_funcs.embeddings_handling import create_indexes_matrix_from_folder, handle_all_with_av, get_max_length
import numpy as np

'''
vecs_file = '/data_repository/data/glove.840B.300d.txt'
all_vecs = read_vecs(vecs_file)

train_dir_path = '/home/dpappas/PyCharm_Projects/youtube_simplest_CNN_2/data/training/'
test_dir_path = '/home/dpappas/PyCharm_Projects/youtube_simplest_CNN_2/data/testing/'

my_words = create_distinct_words_from_folder(train_dir_path , test_dir_path)

my_vecs = get_my_vecs(all_vecs,my_words)

my_voc,my_emb = handle_vecs(my_vecs)

test_indices = create_indexes_matrix_from_folder(my_voc,test_dir_path)
train_indices = create_indexes_matrix_from_folder(my_voc,train_dir_path)

pickle.dump( test_indices, open( "test_indices.p", "wb" ) )
pickle.dump( train_indices, open( "train_indices.p", "wb" ) )
pickle.dump( my_vecs, open( "my_vecs.p", "wb" ) )
pickle.dump( my_voc, open( "my_voc.p", "wb" ) )
pickle.dump( my_emb, open( "my_emb.p", "wb" ) )
pickle.dump( my_words, open( "my_words.p", "wb" ) )
pickle.dump( all_vecs, open( "all_vecs.p", "wb" ) )


my_vecs = pickle.load( open( dir_path+"my_vecs.p", "rb" ) )
my_voc = pickle.load( open( dir_path+"my_voc.p", "rb" ) )
my_emb = pickle.load( open( dir_path+"my_emb.p", "rb" ) )
my_words = pickle.load( open( dir_path+"my_words.p", "rb" ) )
all_vecs = pickle.load( open( dir_path+"all_vecs.p", "rb" ) )

'''

dir_path = '/data_repository/Common_Crawl_840B_tokkens_pickles/'

test_indices = pickle.load( open( dir_path+"test_indices.p", "rb" ) )
train_indices = pickle.load( open( dir_path+"train_indices.p", "rb" ) )

scores = pd.read_csv('/data_repository/data/Personality_scores.csv',index_col= 0,header = 0, delimiter=' ')

av = pd.read_csv('/data_repository/data/YouTube-Personality-audiovisual_features.csv',index_col= 0,header = 0, delimiter=' ')

metr = get_max_length(test_indices,train_indices)

test = handle_all_with_av(scores, test_indices, metr, av)
train = handle_all_with_av(scores, train_indices, metr, av)

pickle.dump( test, open( "test.p", "wb" ) )
pickle.dump( train, open( "train.p", "wb" ) )













