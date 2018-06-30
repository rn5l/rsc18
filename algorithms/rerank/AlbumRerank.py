'''
Created on 17.04.2018
@author: malte
'''

import numpy as np
import pandas as pd

class AlbumRerank:
    
    def __init__(self, base, factor=None, return_num_preds=500 ):
        self.base = base
        self.factor = factor
        self.return_num_preds = return_num_preds
        
    def train(self, data, test=None):
        self.base.train(data, test=test)
        
        actions = data['actions']
        tracks = data['tracks']
        tracks['album_id'] = tracks['album_uri'].astype('category').cat.codes
        actions = pd.merge( actions, tracks[['track_id','album_id']], on='track_id', how='inner' )
        
        self.album_map = pd.DataFrame()
        self.album_map['album_id'] = actions.groupby( 'track_id' ).album_id.min()
        self.album_map = self.artist_map['album_id'].to_dict()
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        baseRes = self.base.predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden ).copy()
        baseRes['confidence'] = ( baseRes['confidence'] - baseRes['confidence'].min() ) / (baseRes['confidence'].max() - baseRes['confidence'].min() )
        #baseRes['confidence'] = baseRes['confidence'] / baseRes['confidence'].max()
        
        if tracks is not None and len(tracks) > 0:
            
            albums = self.album_map['album_id'][tracks].values
            
            y = np.bincount(albums)
            ii = np.nonzero(y)[0]
            freq = pd.Series( index=ii, data=y[ii] )
            
            baseRes['add'] = baseRes.track_id.map( lambda t : freq[self.album_map[t]] if self.album_map[t] in freq else 0 )
             
            if self.factor is None:
                baseRes['confidence'] = baseRes['confidence'].values + baseRes['add']
            else:
                baseRes['confidence'] += baseRes['confidence'] * ( baseRes['add'] / len(freq) ) * self.factor
                                    
            baseRes.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )    
                
        return baseRes.head(self.return_num_preds)
    
    
    