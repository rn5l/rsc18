'''
Created on 17.04.2018

@author: malte
'''
import pandas as pd

class RandomRerank:
    
    def __init__(self, base, only_hidden=False, hidden_fact=1, base_preds = 500, return_num_preds=500 ):
        self.base = base
        self.return_num_preds = return_num_preds
        self.base_preds = base_preds
        self.only_hidden = only_hidden
        self.hidden_fact = hidden_fact
        
    def train(self, data, test=None):
        self.base.return_num_preds = self.base_preds
        self.base.train(data, test=test)
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        base_res = self.base.predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden ).copy()
        
        if self.only_hidden: 
            num = num_hidden * self.hidden_fact
            if num < self.base_preds:
                base_res_short = base_res.head(num)
            else:
                num = self.base_preds
                base_res_short = base_res
                
            base_res = pd.concat([base_res_short.sample(frac=1),base_res.tail( len(base_res)-num )] )
            return base_res.head(self.return_num_preds)  
        else:
            return base_res.sample(frac=1).head(self.return_num_preds)    