'''
Created on 17.04.2018
@author: malte
'''
import pandas as pd

class Cascading:
    
    def __init__(self, first, second, base_preds=1000, factor=1, return_num_preds=500, only_hidden=False, hidden_fact=1 ):
        self.first = first
        self.second = second
        self.base_preds = base_preds
        self.return_num_preds = return_num_preds
        self.only_hidden = only_hidden
        self.hidden_fact = hidden_fact
        
    def train(self, data, test=None):
                
        self.first.return_num_preds = self.base_preds
        self.second.return_num_preds = data['actions']['track_id'].nunique()
        
        self.first.train(data, test=test)
        self.second.train(data, test=test)
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        base_res = self.first.predict( name, tracks, playlist_id=playlist_id, artists=artists )
        casc_res = self.second.predict( name, tracks, playlist_id=playlist_id, artists=artists )
        casc_res['nuc'] = casc_res['confidence']
        del casc_res['confidence']
        
        if self.only_hidden: 
            num = num_hidden * self.hidden_fact
            if num < self.base_preds:
                base_res_short = base_res.head(num)
            else:
                num = self.base_preds
                base_res_short = base_res
        else:
            num = self.base_preds
            base_res_short = base_res
        
        base_res_short = base_res_short.merge( casc_res, how='left', on='track_id' )
        
        base_res_short['confidence'] = base_res_short['confidence'] * 1e-8 + base_res_short['confidence'].max()
        base_res_short['nuc'] = base_res_short['nuc'].fillna(0)
        base_res_short['confidence'] = base_res_short['confidence'] + base_res_short['nuc']
        del base_res_short['nuc']
        
        base_res_short = pd.concat( [base_res_short, base_res.tail(self.base_preds-num)] )
        
        base_res_short.sort_values( ['confidence','track_id'], ascending=False, inplace=True )
        
        return base_res_short.head(self.return_num_preds)
    