# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: malte
"""

import numpy as np
import pandas as pd
from math import log10
import collections as col
import time
import os
import pickle

class SequentialRules: 
    '''
    AssosiationRules(steps=10, weighting='div', tr_steps=10, tr_weighting='div', session_key='playlist_id', item_key='track_id', folder=None, return_num_preds=500 )
        
    Parameters
    --------
    
    '''
    
    def __init__( self, steps=100, weighting='div', tr_steps=100, tr_weighting='div', session_key='playlist_id', item_key='track_id', folder=None, return_num_preds=500 ):
        
        self.steps = steps
        self.weighting = weighting
        self.tr_steps = tr_steps
        self.tr_weighting = tr_weighting
        self.pruning = return_num_preds + 100
        self.session_key = session_key
        self.item_key = item_key
        self.return_num_preds = return_num_preds

        self.session = -1
        self.session_items = []
        
        self.folder = folder
            
    def train(self, data, test=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
        
            
        '''
        
        train = data['actions']
        #playlists = data['playlists']
        testitems = set( test['actions']['track_id'].unique() )

        cur_session = -1
        last_items = []
        rules = dict()
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        cnt = 0
        tstart = time.time()
        
        folder = self.folder
        name = 'sr_rules_'+self.tr_weighting+'_'+str(self.tr_steps)+'.pkl'
        
        if folder is not None and os.path.isfile( folder + name ):
            rules = pickle.load( open( folder + name, 'rb') )
        else: 
        
            for row in train.itertuples( index=False ):
                
                session_id, item_id = row[index_session], row[index_item]
                            
                if session_id != cur_session:
                    cur_session = session_id
                    last_items = []
                else: 
                    cut = len( set(last_items) & testitems ) > 0                    
                    if cut:
                        for i in range( 1, self.tr_steps+1 if len(last_items) >= self.tr_steps else len(last_items)+1 ):
                            prev_item = last_items[-i]
                            
                            if not prev_item in rules :
                                rules[prev_item] = dict()
                            
                            if not item_id in rules[prev_item]:
                                rules[prev_item][item_id] = 0
                            
                            rules[prev_item][item_id] += getattr(self, self.tr_weighting)( i )
                        
                last_items.append(item_id)
                
                cnt += 1
                    
                if cnt % 100000 == 0:
                    print( ' -- finished {} of {} rows in {}s'.format( cnt, len(train), (time.time() - tstart) ) )
            
            if folder is not None:
                pickle.dump( rules,  open( folder + name, 'wb' ) )
        
        if self.pruning > 0 :
            self.prune( rules )
        
        print( ' -- finished training in {}s'.format( (time.time() - tstart) ) )

        
        self.rules = rules
    
    def linear(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same(self, i):
        return 1
    
    def div(self, i):
        return 1/i
    
    def log(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic(self, i):
        return 1/(i*i)
    
    
    def predict( self, plname, tracks, playlist_id=None, artists=None, num_hidden=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        plname : string
            The session IDs of the event.
        tracks : list of int
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        #items = tracks.track_id.values    
        items = tracks if tracks is not None else []   
        
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] = []
        res_dict['confidence'] = []
        
        if len(items) > 0 and items[-1] in self.rules:
            res_dict['track_id'] = [x for x in self.rules[items[-1]]]
            res_dict['confidence'] = [self.rules[items[-1]][x] for x in self.rules[items[-1]]]
            res = pd.DataFrame.from_dict(res_dict)
            res_dict.clear()
            
            res['confidence'] = res['confidence'] / res['confidence'].sum()
            
            if self.steps is not None:
            
                for i in range( self.steps ):
                    
                    if len( items ) >= i + 2:
                        prev = items[ -(i+2) ]
                        
                        res_dict['track_id'] = [x for x in self.rules[prev]]
                        res_dict['tmp'] = [self.rules[prev][x] for x in self.rules[prev]]
                        tmp = pd.DataFrame.from_dict(res_dict)
                        
                        res = res.merge( tmp, how="left", on='track_id' )
                        res['confidence'] += getattr(self, self.weighting + '_pred' )( res['tmp'].fillna(0), i + 2 )
                        del res['tmp']
                        
                        mask = ~np.in1d( tmp.track_id, res['track_id'] )
                        if mask.sum() > 0:
                            tmp['confidence'] = getattr(self, self.weighting + '_pred' )( tmp['tmp'], i + 2 )
                            del tmp['tmp']
                            res = pd.concat( [ res, tmp[mask] ] )
                            
            
            res.sort_values('confidence', ascending=False, inplace=True)
            
            res = res[ np.in1d(res.track_id, tracks, invert=True) ]
            
            return res.head(self.return_num_preds)
        
        return pd.DataFrame.from_dict(res_dict)
    
    def prune(self, rules): 
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
        Parameters
            --------
            rules : dict of dicts
                The rules mined from the training data
        '''
        for k1 in rules:
            tmp = rules[k1]
            if self.pruning < 1:
                keep = len(tmp) - int( len(tmp) * self.pruning )
            elif self.pruning >= 1:
                keep = self.pruning
            counter = col.Counter( tmp )
            rules[k1] = dict()
            for k2, v in counter.most_common( keep ):
                rules[k1][k2] = v             
                
                
    def same_pred(self, confidences, step):
        return confidences
    
    def div_pred(self, confidences, step):
        return confidences / step
    
    def log_pred(self, confidences, step):
        return confidences/(log10(step+1.7))
                
                
     
