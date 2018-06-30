'''
Created on 17.04.2018

@author: malte
'''
import pandas as pd

class PopRerank:
    
    def __init__(self, base, factor=1, return_num_preds=500 ):
        self.base = base
        self.factor = factor
        self.return_num_preds = return_num_preds
        
    def train(self, data, test=None):
        self.base.train(data, test=test)
        
        self.pop_map = pd.DataFrame()
        self.pop_map['pop'] = data['actions'].groupby('track_id').size()
        self.pop_map['pop'] = self.pop_map['pop'] / self.pop_map['pop'].sum()
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None):
        
        base_res = self.base.predict( name, tracks, playlist_id=playlist_id, artists=artists )
        
        base_res['confidence'] += self.pop_map['pop'][base_res.track_id].values * self.factor
        
        base_res.sort_values( ['confidence','track_id'], ascending=False, inplace=True )
        
        return base_res.head(self.return_num_preds)
    