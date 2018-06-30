# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: Balázs Hidasi
Based on https://github.com/hidasib/GRU4Rec
Extended to suit the framework
"""

import numpy as np
import pandas as pd
import pickle
import os
import time
from math import log10


class ItemKNN:
    '''
    ItemKNN(n_sims = 100, lmbd = 20, alpha = 0.5, session_key = 'SessionId', item_key = 'ItemId', time_key = 'Time')
    
    Item-to-item predictor that computes the the similarity to all items to the given item.
    
    Similarity of two items is given by:
    
    .. math::
        s_{i,j}=\sum_{s}I\{(s,i)\in D & (s,j)\in D\} / (supp_i+\\lambda)^{\\alpha}(supp_j+\\lambda)^{1-\\alpha}
        
    Parameters
    --------
    n_sims : int
        Only give back non-zero scores to the N most similar items. Should be higher or equal than the cut-off of your evaluation. (Default value: 100)
    lmbd : float
        Regularization. Discounts the similarity of rare items (incidental co-occurrences). (Default value: 20)
    alpha : float
        Balance between normalizing with the supports of the two items. 0.5 gives cosine similarity, 1.0 gives confidence (as in association rules).
    session_key : string
        header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        header of the timestamp column in the input file (default: 'Time')
    
    '''    
    
    def __init__(self, n_sims = 1500, lmbd = 20, alpha = 0.5, steps=100, remind=False, weighting='same', idf_weight=None, pop_weight=None, session_key = 'playlist_id', item_key = 'track_id', time_key = 'pos', folder=None, return_num_preds=500 ):

        self.n_sims = n_sims
        self.lmbd = lmbd
        self.alpha = alpha
        self.steps = steps
        self.weighting = weighting
        self.idf_weight = idf_weight
        self.pop_weight = pop_weight
        self.remind = remind
        self.item_key = item_key
        self.session_key = session_key
        self.time_key = time_key
        self.folder = folder
        self.return_num_preds = return_num_preds


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
        test_items = set( datat[self.item_key].unique() )
        
        folder = self.folder
        
        name = folder + 'iknn_sims.pkl'
        if self.alpha != 0.5:
            name += '.'+str(self.alpha)
        if self.lmbd != 20:
            name += '.'+str(self.lmbd)
        
        if folder is not None and os.path.isfile( name ):
            self.sims = pickle.load( open( name, 'rb') )
        else:
            data.set_index(np.arange(len(data)), inplace=True)
            itemids = data[self.item_key].unique()
            n_items = len(itemids) 
            data = pd.merge(data, pd.DataFrame({self.item_key:itemids, 'ItemIdx':np.arange(len(itemids))}), on=self.item_key, how='inner')
            sessionids = data[self.session_key].unique()
            data = pd.merge(data, pd.DataFrame({self.session_key:sessionids, 'SessionIdx':np.arange(len(sessionids))}), on=self.session_key, how='inner')
            supp = data.groupby('SessionIdx').size()
            session_offsets = np.zeros(len(supp)+1, dtype=np.int32)
            session_offsets[1:] = supp.cumsum()
            index_by_sessions = data.sort_values(['SessionIdx', self.time_key]).index.values
            supp = data.groupby('ItemIdx').size()
            item_offsets = np.zeros(n_items+1, dtype=np.int32)
            item_offsets[1:] = supp.cumsum()
            index_by_items = data.sort_values(['ItemIdx', self.time_key]).index.values
            self.sims = dict()
            
            cnt = 0
            tstart = time.time()
            
            for i in range(n_items):
                
                if itemids[i] not in test_items:
                    continue
                
                iarray = np.zeros(n_items)
                start = item_offsets[i]
                end = item_offsets[i+1]
                for e in index_by_items[start:end]:
                    uidx = data.SessionIdx.values[e]
                    ustart = session_offsets[uidx]
                    uend = session_offsets[uidx+1]
                    user_events = index_by_sessions[ustart:uend]
                    iarray[data.ItemIdx.values[user_events]] += 1
                iarray[i] = 0
                norm = np.power((supp[i] + self.lmbd), self.alpha) * np.power((supp.values + self.lmbd), (1.0 - self.alpha))
                norm[norm == 0] = 1
                iarray = iarray / norm
                indices = np.argsort(iarray)[-1:-1-self.n_sims:-1]
                self.sims[itemids[i]] = pd.Series(data=iarray[indices], index=itemids[indices])
                
                cnt += 1
                
                if cnt % 1000 == 0:
                    print( ' -- finished {} of {} items in {}s'.format( cnt, len(test_items), (time.time() - tstart) ) )
                
            if folder is not None:
                pickle.dump( self.sims, open( name, 'wb') )
        
        if self.idf_weight != None:
            self.idf = pd.DataFrame()
            self.idf['idf'] = data.groupby( self.item_key ).size()
            self.idf['idf'] = np.log( data[self.session_key].nunique() / self.idf['idf'] )
            self.idf['idf'] = ( self.idf['idf'] - self.idf['idf'].min() ) / ( self.idf['idf'].max() - self.idf['idf'].min() )
            self.idf = pd.Series( index=self.idf.index, data=self.idf['idf']  )
            
        if self.pop_weight != None:
            self.pop = pd.DataFrame()
            self.pop['pop'] = data.groupby( self.item_key ).size()
            self.pop['pop'] = ( self.pop['pop'] - self.pop['pop'].min() ) / ( self.pop['pop'].max() - self.pop['pop'].min() )
            self.pop = pd.Series( index=self.pop.index, data=self.pop['pop']  )
        
            
    def predict( self, name=None, tracks=None, artists=None, playlist_id=None, num_hidden=None ):
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
        if len(items) > 0 and items[-1] in self.sims:
            sim_list = self.sims[items[-1]]
        
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] =  sim_list.index
        res_dict['confidence'] = sim_list
        if len(items) > 0 and self.idf_weight != None:
            res_dict['confidence'] = res_dict['confidence'] * self.idf[ items[-1] ]
        res = pd.DataFrame.from_dict(res_dict)
        
        if self.steps is not None:
            
            if len( items ) < 10:
                self.step_size = 0.1
            else:
                self.step_size = 1.0 / (len( items ) + 1)
            
            for i in range( self.steps ):
                
                if len( items ) >= i + 2:
                    prev = items[ -(i+2) ]
                    
                    if prev not in self.sims:
                        continue
                    
                    sim_list = self.sims[prev]
                    
                    res = res.merge( sim_list.to_frame('tmp'), how="left", left_on='track_id', right_index=True )
                    if self.idf_weight != None:
                        res['tmp'] = res['tmp'] * self.idf[ prev ]
                    if self.pop_weight != None:
                        res['tmp'] = res['tmp'] * self.pop[ prev ] # * (1 - self.pop[ prev ])
                    res['confidence'] += getattr(self, self.weighting)( res['tmp'].fillna(0), i + 2 )
                    
                    #res['confidence'] += res['tmp'].fillna(0)
                    del res['tmp']
                    
                    mask = ~np.in1d( sim_list.index, res['track_id'] )
                    if mask.sum() > 0:
                        res_add = {}
                        res_add['track_id'] =  sim_list[mask].index
                        if self.idf_weight != None:
                            res_add['confidence'] = sim_list[mask] * self.idf[ prev ]
                        else:
                            res_add['confidence'] = sim_list[mask]
                        res_add['confidence'] = getattr(self, self.weighting)( res_add['confidence'], i + 2 )
                        #res_add['confidence'] = sim_list[mask]
                        res_add = pd.DataFrame.from_dict(res_add)
                        res = pd.concat( [ res, res_add ] )
        
        if not self.remind:
            res = res[ np.in1d( res.track_id, items, invert=True ) ]
        
        res.sort_values( 'confidence', ascending=False, inplace=True )
        
        #if self.normalize:
        #    res['confidence'] = res['confidence'] / res['confidence'].sum()
        
        res=res.head(self.return_num_preds) 
        
        return res
    
    
    def same(self, confidences, step):
        return confidences
    
    def div(self, confidences, step):
        return confidences / step
    
    def log(self, confidences, step):
        return confidences/(log10(step+1.7))
    
    def linear(self, confidences, step):
        return confidences * (1 - (self.step_size * step))
    
    def set_return_num_preds(self, num):
        self.return_num_preds = num
