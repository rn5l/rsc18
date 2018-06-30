# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: malte
"""

import numpy as np
import pandas as pd
from scipy import sparse
import implicit

class ImplicitNN:
    '''
    ImplicitNN(factors=100, epochs=15, reg=0.03, steps=None, weighting='same', session_key = 'playlist_id', item_key = 'track_id')
          
    Using the implicit library to find nearest neighbors in terms of the items or tracks respectively
    
    Parameters
    --------
    
    '''
    def __init__(self, factors=100, epochs=15, reg=0.03, steps=None, weighting='same', session_key = 'playlist_id', item_key = 'track_id'):
        self.factors = factors
        self.epochs = epochs
        self.reg = reg
        self.steps = steps
        self.weighting = weighting

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
        
        #self.model = implicit.als.AlternatingLeastSquares( factors=self.factors, iterations=self.epochs, regularization=self.reg )
        self.model = implicit.approximate_als.NMSLibAlternatingLeastSquares( factors=self.factors, iterations=self.epochs, regularization=self.reg )
        #self.model = implicit.nearest_neighbours.CosineRecommender()
        self.model.fit(self.mat)
        
        self.tmp = self.mat.T.tocsr()
    
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
        
        sim_list = pd.Series()
        if len(items) > 0:
            sim_list = self.model.similar_items( self.itemidmap[items[-1]] , N=1000)
        
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] =  self.itemidmap2[ [e[0] for e in sim_list ] ]
        res_dict['confidence'] = [e[1] for e in sim_list ]
        res = pd.DataFrame.from_dict(res_dict)
        
        if self.steps is not None:
            
            for i in range( self.steps ):
                
                if len( items ) >= i + 2:
                    prev = items[ -(i+2) ]
                    
                    sim_list = self.model.similar_items( self.itemidmap[prev] , N=1000)
                    sim_list = pd.Series( index=self.itemidmap2[ [e[0] for e in sim_list ] ], data=[e[1] for e in sim_list ] )
                    
                    res = res.merge( sim_list.to_frame('tmp'), how="left", left_on='track_id', right_index=True )
                    res['confidence'] += res['tmp'].fillna(0)
                    del res['tmp']
                    
                    mask = ~np.in1d( sim_list.index, res['track_id'] )
                    if mask.sum() > 0:
                        res_add = {}
                        res_add['track_id'] =  sim_list[mask].index
                        res_add['confidence'] = getattr(self, self.weighting)( sim_list[mask], i + 2 )
                        res_add = pd.DataFrame.from_dict(res_add)
                        res = pd.concat( [ res, res_add ] )
           
        
        res = res[ ~np.in1d( res.track_id, items ) ]
        res.sort_values( 'confidence', ascending=False, inplace=True )
        
        return res.head(500)
    
    def same(self, confidences, step):
        return confidences
    
    def div(self, confidences, step):
        return confidences / step
    