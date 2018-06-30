'''
Created on 13.04.2018
@author: malte
'''
from algorithms.Model import Model
import pandas as pd
import numpy as np
import time

class MostPopular(Model):
    '''
    classdocs
    '''
    
    def __init__(self, remind=False, return_num_preds=500 ):
        self.return_num_preds = return_num_preds
        self.remind = remind
        
    def init(self, train, test):
        pass
    
    def train(self, train, test=None):
        
        print( 'training mostpopular' )
        
        tstart = time.time()
        
        pop = pd.DataFrame()
        pop['popularity'] = train['actions'].groupby( 'track_id' ).size()
        pop.reset_index(inplace=True)
        pop['confidence'] = pop['popularity'] / pop['popularity'].max()
        #del pop['popularity']
        pop.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
        #self.pop = pop.head(self.return_num_preds + 100)[['track_id','confidence']] # not sure why this fails when not having "+ 100" in head()
        self.pop = pop[['track_id','confidence']] 
        
        print( ' -- finished training in {}s'.format( (time.time() - tstart) ) )
            
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        items = tracks if tracks is not None else []
        if self.remind:
            res = self.pop.head(self.return_num_preds)
        else:
            res = self.pop[ np.in1d( self.pop.track_id, items, invert=True ) ].head(self.return_num_preds)
        return res
        