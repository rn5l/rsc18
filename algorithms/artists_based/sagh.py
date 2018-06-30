'''
Created on 17.04.2018
@author: malte
'''
import numpy as np
import pandas as pd

class SAGH:
    
    def __init__(self, normalize=False, item_key='track_id', artist_key='artist_id', session_key='playlist_id', return_num_preds=500):
        self.item_key = item_key
        self.artist_key = artist_key
        self.session_key = session_key
        
        self.normalize = normalize
        self.return_num_preds = return_num_preds
        
    def train(self, data, test=None):
        
        train = data['actions']
        
        agh = pd.DataFrame()
        agh['apop'] = train.groupby( [self.artist_key, self.item_key] ).size()
        if self.normalize:
            agh = agh.reset_index()
            agh['apop'] = agh['apop'] / agh.groupby( [self.artist_key] )['apop'].transform('sum')
            print(agh)
            agh = agh
            #agh = agh.reset_index()
            self.agh = pd.DataFrame()
            self.agh = agh.groupby( [self.artist_key, self.item_key] ).mean()
        else: 
            self.agh = agh
            
        pop = pd.DataFrame()
        pop['pop'] = train.groupby( [self.item_key] ).size()
        pop['pop'] = pop['pop'] / len(train)
        self.pop = pop
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
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
        res = pd.DataFrame()
        
        if artists is None or len(artists) == 0:
            res['track_id'] = []
            res['confidence'] = []
            return res
        
        idxs = pd.IndexSlice
        sagh = self.agh.loc[ idxs[ artists,: ] ]
        sagh = sagh.reset_index()
        res['confidence'] = sagh.groupby( self.item_key )['apop'].sum()
        res.reset_index(inplace=True)
        res['confidence'] += self.pop['pop'][ res[self.item_key].values ].values
        res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
        
        return res.head(500)
    