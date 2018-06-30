'''
Created on 10.10.2017
@author: malte
'''
import time

import implicit

from helper import inout
import numpy as np
import pandas as pd
import scipy.sparse as sparse

FOLDER_TRAIN = '../../../data/data_formated_50k/'
FOLDER_TEST = '../../../data/sample_50k_similar/'

FIELD_USER = 'playlist_id'

def main():
    train, test = inout.load_dataset( FOLDER_TRAIN, FOLDER_TEST, feather=True )
    
    actions = train['actions']
    actions = pd.concat( [ test['actions'], actions ] ) 
    
    create_latent_factors( actions, size=32, iterations=20, count=False, save=FOLDER_TEST )
    create_latent_factors( actions, size=32, iterations=20, count=True, item='artist_id', save=FOLDER_TEST )
    

def create_latent_factors( combi, size=10, iterations=10, count=False, user='playlist_id', item='track_id', save='' ):
    
    start = time.time()
    
    combi = combi[ [ user, item ] ]
    if count: 
        cn = pd.DataFrame()
        cn['value'] = combi.groupby( [user,item] ).size()
        combi = cn.reset_index()
    else:
        combi.drop_duplicates( keep='first' )
        combi['value'] = 1.0
    
    umap = pd.Series( index=combi[user].unique(), data=range(combi[user].nunique()) )
    imap = pd.Series( index=combi[item].unique(), data=range(combi[item].nunique()) )
    
    SPM = sparse.csr_matrix(( combi['value'].tolist(), (imap[ combi[item].values ], umap[ combi[user].values ] )), shape=( combi[item].nunique(), combi[user].nunique() ))
    
    print( 'created user features in ',(time.time() - start) )
    
    start = time.time()
    
    model = implicit.als.AlternatingLeastSquares( factors=size, iterations=iterations )

    # train the model on a sparse matrix of item/user/confidence weights
    model.fit(SPM)
    
    Ufv = model.user_factors
    Ifv =  model.item_factors
    
    UF = ['lf_'+str(i) for i in range(size)]
    SF = ['lf_'+str(i) for i in range(size)]
    
    Uf = pd.DataFrame( Ufv, index=umap.index )
    Uf.columns = UF
    Uf[user] = Uf.index
    
    If = pd.DataFrame( Ifv, index=imap.index )
    If.columns = SF
    If[item] = If.index
     
    Uf.to_csv(save+'als_'+user+'_features'+('_cnt' if count else '')+'.'+str(size)+'.csv', index=False)
    If.to_csv(save+'als_'+item+'_features'+('_cnt' if count else '')+'.'+str(size)+'.csv', index=False)
    
    print('created latent social features in ',(time.time() - start))
    
    res = []
    
    print(len(Ifv))
    print(len(combi[item].unique()) )
    print(len(Ufv))
    print(len(combi[user].unique()) )
          
    for row in combi.itertuples(index=False):
        
        songv = Ifv[imap[row[1]]]
        userv = Ufv[umap[row[0]]]
        
        res.append( np.dot( userv, songv.T ) )
        
    combi['reconst'] = res
    
    print( combi[[user,item,'value','reconst']] )
  
if __name__ == '__main__':
    
    main()
    
    