# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:57:27 2015
@author: mludewig
"""

import numpy as np
import pandas as pd
from math import log10
import collections as col
import time
import os
import pickle

class AssosiationRules: 
    '''
    AssosiationRules(steps=None, weighting='same', session_key='playlist_id', item_key='track_id', folder=None, return_num_preds=500)
        
    Parameters
    --------
    
    '''
    
    def __init__( self, steps=100, weighting='same', session_key='playlist_id', item_key='track_id', folder=None, return_num_preds=500  ):
        
        self.steps = steps
        self.weighting = weighting
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
        test: pandas.DataFrame
            
        '''
        
        train = data['actions']
        testitems = set( test['actions']['track_id'].unique() )
        #playlists = data['playlists']
        
        cur_session = -1
        last_items = set()
        rules = dict()
        
        index_session = train.columns.get_loc( self.session_key )
        index_item = train.columns.get_loc( self.item_key )
        
        cnt = 0
        tstart = time.time()
        
        folder = self.folder
        
        if folder is not None and os.path.isfile( folder + 'ar_rules.pkl' ):
            rules = pickle.load( open( folder + 'ar_rules.pkl', 'rb') )
        else: 
        
            for row in train.itertuples( index=False ):
                
                session_id, item_id = row[index_session], row[index_item]
                
                if session_id != cur_session:
                    cur_session = session_id
                    last_items = set()
                else:
                    cut = len( last_items & testitems ) > 0
                    actual = item_id in testitems
                    
                    if cut or actual:
                        for item_id2 in last_items: 
                            
                            if not item_id in rules :
                                rules[item_id] = dict()
                            
                            if not item_id2 in rules :
                                rules[item_id2] = dict()
                            
                            if not item_id in rules[item_id2]:
                                rules[item_id2][item_id] = 0
                            
                            if not item_id2 in rules[item_id]:
                                rules[item_id][item_id2] = 0
                            
                            rules[item_id][item_id2] += 1
                            rules[item_id2][item_id] += 1
                        
                last_items.add( item_id )
                
                cnt += 1
                    
                if cnt % 100000 == 0:
                    print( ' -- finished {} of {} rows in {}s'.format( cnt, len(train), (time.time() - tstart) ) )
        
            if folder is not None:
                pickle.dump( rules,  open( folder + 'ar_rules.pkl', 'wb' ) )
        
        if self.pruning > 0 :
            self.prune( rules )
            
        print( ' -- finished training in {}s'.format( (time.time() - tstart) ) )
            
        self.rules = rules
        
    def predict( self, plname, tracks, playlist_id=None, artists=None, num_hidden=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : string
            playlist name
        tracks : int list
            tracks in the current playlist
        playlist_id : int
            playlist identifier
        artists : int list
            artists in the current playlist (per track)
        num_hidden : int
            Number of hidden tracks in the list
            
        Returns
        --------
        res : pandas.DataFrame
            sorted predictions with track_id and confidence
        
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
                        res['confidence'] += getattr(self, self.weighting )( res['tmp'].fillna(0), i + 2 )
                        del res['tmp']
                        del tmp['tmp']
                        
                        mask = ~np.in1d( tmp.track_id, res['track_id'] )
                        if mask.sum() > 0:
                            tmp['confidence'] = getattr(self, self.weighting )( tmp['tmp'], i + 2 )
                            del tmp['tmp']
                            res = pd.concat( [ res, tmp[mask] ] )
            
            res = res[ np.in1d(res.track_id, tracks, invert=True) ]
            
            res.sort_values('confidence', ascending=False, inplace=True)
            
            return res.head(500)
        
        return pd.DataFrame.from_dict( res_dict )
    
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
    
    def same(self, confidences, step):
        return confidences
    
    def div(self, confidences, step):
        return confidences / step
    
    def log(self, confidences, step):
        return confidences/(log10(step+1.7))
  
