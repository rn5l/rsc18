'''
Created on 17.04.2018

@author: malte
'''
import pandas as pd


class DiversityRerank:
    
    def __init__(self, base, only_hidden=False, return_num_preds=500 ):
        self.base = base
        self.return_num_preds = return_num_preds
        self.only_hidden = only_hidden
        
    def train(self, data, test=None):
        self.base.train(data, test=test)
        self.artist_map = pd.DataFrame()
        self.artist_map['artist_id'] = data['actions'].groupby('track_id').artist_id.min()
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        baseRes = self.base.predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden ).copy()
        baseRes['artist_id'] = baseRes.track_id.map( lambda t : self.artist_map['artist_id'][t] )
        
        artist_in = {}
        add = []
        
        i = 0
        for row in baseRes.itertuples():
            i += 1
            if self.only_hidden and i > num_hidden:
                add.append(0)
                continue
            
            if not row.artist_id in artist_in:
                add.append( 1 )
                artist_in[row.artist_id] = 1
            else:
                if artist_in[row.artist_id] < 1:
                    add.append( 1 )
                else:
                    add.append( 0 )
                    artist_in[row.artist_id] += 1
                
        baseRes['add'] = add
        baseRes['confidence'] = ( baseRes['confidence'] - baseRes['confidence'].min() ) / ( baseRes['confidence'].max() - baseRes['confidence'].min() )
        baseRes['confidence'] += baseRes['add']
        
        baseRes.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
                
        return baseRes.head(self.return_num_preds)
    