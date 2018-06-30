# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015

@author: Malte
"""

import numpy as np
import pandas as pd
from scipy import sparse
import implicit

class ColdImplicit:
    '''
    ColdImplicit( n_factors = 100, epochs = 10, lr = 0.01, reg=0.01, algo='als', idf_weight=False, session_key = 'playlist_id', item_key = 'track_id')
            
    Parameters
    --------
    
    '''
    def __init__(self, n_factors = 100, epochs = 10, lr = 0.01, reg=0.01, algo='als', idf_weight=False, session_key = 'playlist_id', item_key = 'track_id'):
        self.factors = n_factors
        self.epochs = epochs
        self.lr = lr
        self.reg = reg
        
        self.algo = algo
        self.idf_weight = idf_weight
        self.session_key = session_key
        self.item_key = item_key
        self.current_session = None
    
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
        
        data = pd.concat([data,datat])
        
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        self.itemidmap2 = pd.Series(index=np.arange(self.n_items), data=itemids)
        
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.useridmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        
        ones = np.ones( len(data) )
        
        row_ind = self.itemidmap[ data.track_id.values ]
        col_ind = self.useridmap[ data.playlist_id.values ] 
        
        self.mat = sparse.csr_matrix((ones, (row_ind, col_ind)))
        
        if self.algo == 'als':
            self.model = implicit.als.AlternatingLeastSquares( factors=self.factors, iterations=self.epochs, regularization=self.reg )
        elif self.algo == 'bpr': 
            self.model = implicit.bpr.BaysianPersonalizedRanking( factors=self.factors, iterations=self.epochs, regularization=self.reg )
                
        self.model.fit(self.mat)
                
        self.tmp = self.mat.T.tocsr()
        
        if self.idf_weight:
            self.idf = pd.DataFrame()
            self.idf['idf'] = train.groupby( self.item_key ).size()
            self.idf['idf'] = np.log( train[self.session_key].nunique() / self.idf['idf'] )
            self.idf = self.idf['idf'].to_dict()
    
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
        
        items = tracks if tracks is not None else []
        
        if len(items) == 0:
            res_dict = {}
            res_dict['track_id'] = []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)
        
        
        itemidxs = self.itemidmap[items]
        if self.idf_weight:
            weight = self.idf.ix[items]['idf'].values
            factors = self.model.item_factors[itemidxs]
            factors = factors * weight
            
            uF = factors.sum(axis=0) / weight.sum()
        else:
            uF = self.model.item_factors[itemidxs].mean(axis=0)
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] =  self.itemidmap.index
        res_dict['confidence'] = self.model.item_factors.dot(uF)
        res = pd.DataFrame.from_dict(res_dict)
        
        res = res[ ~np.in1d( res.track_id, tracks ) ]
        res.sort_values( 'confidence', ascending=False, inplace=True )
        
        return res.head(500)
    