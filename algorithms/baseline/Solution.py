'''
Created on 13.04.2018
@author: malte
'''
from algorithms.Model import Model
import pandas as pd
import numpy as np
import time

class Solution(Model):
    '''
    classdocs
    '''
    
    def __init__(self, file):
        '''
        Constructor
        '''
        self.file = file;
        
    def init(self, train, test):
        pass
    
    def train(self, train, test=None):
        
        print( 'training solution' )
        
        tstart = time.time()
        
        self.solution = pd.read_csv( self.file )
        
        print( ' -- finished training in {}s'.format( (time.time() - tstart) ) )
            
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        return self.solution[ self.solution.playlist_id == playlist_id ]
