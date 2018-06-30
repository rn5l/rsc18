'''
Created on 17.04.2018

@author: malte
'''
import numpy as np
import pandas as pd

class ArtistRerank:
    
    def __init__(self, base, factor=None, return_num_preds=500 ):
        self.base = base
        self.factor = factor
        self.return_num_preds = return_num_preds
        
    def train(self, data, test=None):
        self.base.train(data, test=test)
        
        self.artist_map = pd.DataFrame()
        self.artist_map['track_id'] = data['actions'].groupby('track_id').artist_id.min()
        self.artist_map = self.artist_map['track_id'].to_dict()
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None):
        
        baseRes = self.base.predict( name, tracks, playlist_id=playlist_id, artists=artists )
        
        if artists is not None:
            
            y = np.bincount(artists)
            ii = np.nonzero(y)[0]
            freq = pd.Series( index=ii, data=y[ii] )
            
            baseRes['add'] = baseRes.track_id.map( lambda t : freq[self.artist_map[t]] if self.artist_map[t] in freq else 0 )
            if self.factor is None:
                baseRes['confidence'] += baseRes['add']
            else:
                baseRes['confidence'] += baseRes['confidence'] * ( baseRes['add'] / len(freq) ) * self.factor
        
            baseRes.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )    
                
        return baseRes.head(self.return_num_preds)
    