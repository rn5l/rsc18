'''
Created on 13.04.2018
@author: malte
'''
from algorithms.Model import Model
import pandas as pd
import numpy as np

class Random(Model):
    '''
    classdocs
    '''
    
    def __init__(self):
        '''
        Constructor
        '''
        
    def init(self, train, test):
        pass
    
    def train(self, train):
        self.tracks = train['tracks']
            
    def predict(self, name=None, tracks=None):
        random = self.tracks.sample(500).reset_index()
        random['confidence'] = ( (-random.index + 500) / np.arange(1,501).sum() )
        return random[['track_id','confidence']]
