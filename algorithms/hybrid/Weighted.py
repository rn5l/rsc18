'''
Created on 14.05.2018

@author: malte
'''

import time

import numpy as np
import pandas as pd

class Weighted:
    
    def __init__(self, algs, weights, training=True, return_num_preds=500):
        
        self.algs = algs
        self.weights = weights
        self.training = training
        self.return_num_preds=return_num_preds
        
        if len( self.algs ) != len( self.weights ):
            raise Exception('weights do not fit')
        
    def train(self, data, test=None):
        
        if self.training:
            for a in self.algs:
                a.train(data, test=test)
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        #print('predict weighted')
        res = pd.DataFrame()
        
        #tstart = time.time()
        
        for i in range( len( self.algs ) ):
            baseRes = self.algs[i].predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden ).copy()
            baseRes['confidence'] = ( baseRes['confidence'] - baseRes['confidence'].min() ) / (baseRes['confidence'].max() - baseRes['confidence'].min() )
            
            if len( res ) is 0:
                res = pd.concat( [ res, baseRes ] )
                res['confidence'] = res['confidence'] * self.weights[i]
            else:
                baseRes['confidence'] = baseRes['confidence'] * self.weights[i]
                
                mask_add = np.in1d( baseRes.track_id, res.track_id )
                #mask_add2 = np.isin( res.track_id, baseRes.track_id )
                baseRes['tmp'] = baseRes['confidence'].values
                res = res.merge( baseRes[['track_id','tmp']][mask_add], on='track_id', how='left' )
                res['confidence'] = res['confidence'] + res['tmp'].fillna(0)
                #res['confidence'][mask_add2] = res['confidence'][mask_add2] + res['tmp'][mask_add2] * self.weights[i]
                del res['tmp']
                del baseRes['tmp']
                
                mask_append = ~mask_add
                res = pd.concat( [ res, baseRes[mask_append] ] )
               
        res.sort_values( 'confidence', ascending=False, inplace=True )
        
        #print( (time.time() - tstart) )
        
        return res.head(self.return_num_preds)
    