'''
Created on 17.04.2018

@author: malte
'''
import numpy as np
import pandas as pd

class Fill:
    
    def __init__(self, base, fill, cut=False, stacked=False, weighted=False):
        self.base = base
        self.fill = fill
        
        self.stacked = stacked
        self.weighted = weighted
        self.cut = cut
        
    def train(self, data, test=None):
        
        if self.stacked:
            self.artistmap = data['actions'].groupby('track_id').artist_id.min()
            self.artistmap = pd.Series( index=self.artistmap.index, data=self.artistmap.values )
        
        self.base.train(data, test=test)
        self.fill.train(data, test=test)
        
        self.relmiss = 0
        self.miss = 0
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        baseRes = self.base.predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden ).copy()
        
        if len(baseRes) > baseRes.track_id.nunique():
            print( type( self.base ) )
            print( baseRes.nunique() )
            raise Exception( 'no unique recs ', type(self.base).__name__ ) 
        
        if self.cut > 1:
            baseRes = baseRes.head(self.cut)
             
        if len( baseRes ) < 500: #need to fill up
            
            if len(baseRes) < num_hidden:
                self.relmiss += 1  
            self.miss += 1
            
            if self.stacked:
                tracks_stack = baseRes.track_id.values[:100]
                artists_stack = self.artistmap[ baseRes.track_id.values[:100] ].values
                
#                 if len(baseRes) >= num_hidden:
#                     tracks_stack = []
#                     artists_stack = []
                    
                fillRes = self.fill.predict( name, tracks_stack, playlist_id=playlist_id, artists=artists_stack, num_hidden=num_hidden )
                
            else:
                fillRes = self.fill.predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden )
                
            need = 500 - len( baseRes )
            
            if self.weighted:
                
                baseRes['confidence'] = ( baseRes['confidence'] - baseRes['confidence'].min() ) / ( baseRes['confidence'].max() - baseRes['confidence'].min() )
                fillRes['confidence'] = ( fillRes['confidence'] - fillRes['confidence'].min() ) / ( fillRes['confidence'].max() - fillRes['confidence'].min() )
                
                baseRes['confidence'] = baseRes['confidence'] * 0.5
                fillRes['confidence'] = fillRes['confidence'] * 0.5
                
                mask_add = np.in1d( fillRes.track_id, baseRes.track_id )
                #mask_add2 = np.isin( res.track_id, baseRes.track_id )
                fillRes['tmp'] = fillRes['confidence']
                baseRes = baseRes.merge( fillRes[['track_id','tmp']][mask_add], on='track_id', how='left' )
                baseRes['confidence'] = baseRes['confidence'] + baseRes['tmp'].fillna(0)
                #res['confidence'][mask_add2] = res['confidence'][mask_add2] + res['tmp'][mask_add2] * self.weights[i]
                del baseRes['tmp']
                del fillRes['tmp']
                
                mask_append = ~mask_add
                baseRes = pd.concat( [ baseRes, fillRes[mask_append] ] )
                
                baseRes.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
                
            else:
                fillRes = fillRes[ ~np.in1d(fillRes.track_id, baseRes.track_id ) ]
                fillRes['confidence'] = ( fillRes['confidence'] - fillRes['confidence'].min() ) / ( fillRes['confidence'].max() - fillRes['confidence'].min() )
                if len(baseRes) > 0:
                    fillRes['confidence'] = fillRes['confidence'] * baseRes['confidence'].min()
                baseRes = pd.concat( [ baseRes, fillRes.head(need) ] )
                                
            if len(baseRes) > baseRes.track_id.nunique():
                raise Exception( 'no unique recs ', type(self.fill).__name__ )
            
        return baseRes.head(500)
    
    

    