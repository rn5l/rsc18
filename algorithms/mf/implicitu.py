# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: malte
"""

import numpy as np
import pandas as pd
from scipy import sparse
import implicit
import time

class Implicit:
    '''
    BPR( n_factors = 100, epochs = 10, lr = 0.001, reg=0.005, filter=None, algo='als', session_key = 'playlist_id', item_key = 'track_id' )
            
    Parameters
    --------
    
    '''
    def __init__(self, n_factors = 100, epochs = 10, lr = 0.001, reg=0.005, filter=None, algo='als', session_key = 'playlist_id', item_key = 'track_id'):
        self.factors = n_factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        
        self.filter = filter 
        
        self.algo = algo
        self.session_key = session_key
        self.item_key = item_key
        self.current_session = None
        
        self.callbacks = []
    
    def train(self, train, test=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        data = train['actions']
        datat = test['actions']
        
        if self.filter is not None:
            data = self.filter_data(data, min_uc=self.filter[0], min_sc=self.filter[1])
        
        data = pd.concat([data,datat])
        
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        self.itemidmap2 = pd.Series(index=np.arange(self.n_items), data=itemids)
        
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.useridmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        
        tstart = time.time()
        
        data = pd.merge(data, pd.DataFrame({self.item_key:self.itemidmap.index, 'ItemIdx':self.itemidmap[self.itemidmap.index].values}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:self.useridmap.index, 'SessionIdx':self.useridmap[self.useridmap.index].values}), on=self.session_key, how='inner')
        
        print( 'add index in {}'.format( (time.time() - tstart) ) )
        
        
        tstart = time.time()
        
        ones = np.ones( len(data) )
        
        row_ind = data.ItemIdx
        col_ind = data.SessionIdx
        
        self.mat = sparse.csr_matrix((ones, (row_ind, col_ind)))
        self.tmp = self.mat.T.tocsr()

        print( 'matrix in {}'.format( (time.time() - tstart) ) )
        
        iters = self.epochs
        if len(self.callbacks) > 0:
            iters = 1
        
        if self.algo == 'als':
            
            self.model = implicit.als.AlternatingLeastSquares( factors=self.factors, iterations=iters, regularization=self.reg )
        elif self.algo == 'bpr': 
            self.model = implicit.bpr.BayesianPersonalizedRanking( factors=self.factors, iterations=iters, regularization=self.reg, learning_rate=self.lr )
        
        start = time.time()
        
        if len(self.callbacks) > 0: 
        
            for j in range( self.epochs ):
                
                self.model.fit(self.mat)
                
                print( 'finished epoch {} in {}s'.format( j, ( time.time() - start ) ) )
                
                for c in self.callbacks:
                    if hasattr(c, 'callback'):
                        getattr(c,'callback')( self )
        else:
            self.model.fit(self.mat)
        
    
    def filter_data(self, data, min_uc=5, min_sc=0):
        # Only keep the triplets for items which were clicked on by at least min_sc users. 
        if min_sc > 0:
            itemcount = data[[self.item_key]].groupby(self.item_key).size()
            data = data[data[self.item_key].isin(itemcount.index[itemcount.values >= min_sc])]
        
        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = data[[self.session_key]].groupby(self.session_key).size()
            data = data[data[self.session_key].isin(usercount.index[usercount.values >= min_uc])]
        
        return data
    
    def predict( self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : int or string
            The session IDs of the event.
        tracks : int list
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        res : pandas.DataFrame
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
#         items = np.array( tracks ) if tracks is not None else np.array( [] )
#         items = items[np.isin(items, self.itemmap.index)]
                  
        if playlist_id is not None:
            
            if playlist_id not in self.useridmap.index:
                res_dict = {}
                res_dict['track_id'] = []
                res_dict['confidence'] = []
                return pd.DataFrame.from_dict(res_dict)
            
            recommendations = self.model.recommend( self.useridmap[playlist_id], self.tmp, N=500 )
            
            
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] =  self.itemidmap2[ [x[0] for x in recommendations] ]
        res_dict['confidence'] = [x[1] for x in recommendations]
        res = pd.DataFrame.from_dict(res_dict)
        res.sort_values( 'confidence', ascending=False, inplace=True )
        
        return res.head(500)
    
    def add_epoch_callback(self, clazz):
        
        self.callbacks.append(clazz)
    