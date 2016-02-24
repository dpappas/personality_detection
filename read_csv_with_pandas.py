

import pandas as pd

#fname = 'C:\\Users\\Dimitris\\Desktop\\PERSONALITY\\youtube-personality\\YouTube-Personality-audiovisual_features.csv'

fname = '/home/dpappas/AVfeats.csv'
av = pd.read_csv( fname, index_col=0 , delim_whitespace=True)

#names = ['VLOG441', 'VLOG431']


names = test['names']
test['AV'] = av.loc[names].values

names = train['names']
train['AV'] = av.loc[names].values


pickle.dump( train, open( "train.p", "wb" ) )
pickle.dump( test, open( "test.p", "wb" ) )



