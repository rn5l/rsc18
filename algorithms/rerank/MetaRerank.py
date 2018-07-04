'''
Created on 17.04.2018

@author: malte
'''
import math

from helper import inout
import numpy as np
import pandas as pd


class MetaRerank:
    
    def __init__(self, base, only_hidden=False, threshold=0.15, meta_folder='data/metadata_combined/', return_num_preds=500 ):
        self.base = base
        self.return_num_preds = return_num_preds
        self.only_hidden = only_hidden
        self.meta_folder = meta_folder
        self.threshold = threshold
        
        #self.fields = ['acousticness','danceability','energy','instrumentalness','loudness','mode','speechiness','tempo','valence']
        self.fields = ['energy','loudness','tempo']
        
    def train(self, data, test=None):
        self.base.train(data, test=test)
                
        track_meta = inout.load_meta_track( self.meta_folder )
        track_meta['spotify_id'] = track_meta['track_id']
        del track_meta['track_id']
        
        tracks = data['tracks']
        tracks['spotify_id'] = tracks['track_uri'].apply( lambda x: x.split(':')[2] )
                
        self.tracks = tracks.merge( track_meta, on='spotify_id', how='left' )
        self.tracks = self.tracks[ self.fields + ['track_id'] ]
        self.tracks['tempo'] = ( self.tracks['tempo'] - self.tracks['tempo'].min() ) / ( self.tracks['tempo'].max() - self.tracks['tempo'].min() )
        self.tracks['loudness'] = ( self.tracks['loudness'] - self.tracks['loudness'].min() ) / ( self.tracks['loudness'].max() - self.tracks['loudness'].min() )
        self.tracks = self.tracks.fillna(0)
        self.tracks.reset_index(inplace=True,drop=True)
        self.trackmap = pd.Series( index=self.tracks['track_id'], data=self.tracks.index )
        self.tracks = self.tracks[self.fields].values
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        baseRes = self.base.predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden ).copy()
        tracks = tracks if tracks is not None else []
        
        baseResShort = baseRes
        num = len(baseRes)
        if self.only_hidden: 
            num = num_hidden
            if num < len(baseRes):
                baseResShort = baseRes.head(num).copy()
                
        
        if len(tracks) > 0:
            
            seedidx = self.trackmap[ tracks ].values
            residx = self.trackmap[ baseResShort.track_id.values ].values.astype('int')
        
            seedmeta = self.tracks[seedidx]
            sstd = seedmeta.std(axis=0) + 1e-7
            smean = seedmeta.mean(axis=0)
            
            if sstd.mean() <= self.threshold:
            
                resmeta = self.tracks[residx]
                
                sim = np.dot( resmeta, smean )
                seednorm = np.sum( np.square( smean ) )
                resnorm = np.sum( np.square( resmeta ),axis=1 )
                sim = sim / ( seednorm * resnorm ) 
                
    #             if len(tracks)  > 1:
    #                 resmeta = resmeta.T
    #                 for i in range(len(self.fields)):
    #                     resmeta[i] = self.normpdf( resmeta[i], smean[i], sstd[i] )
    #                 
    #                 sim = resmeta.sum(axis=0)
    #             else:
    #                 resmeta = np.abs( resmeta - seedmeta )
    #                 sim = (1 - resmeta).sum(axis=1)
    #             
    #             sim = (sim - sim.min()) / (sim.max() - sim.min()) 
                            
                baseResShort['confidence'] += baseResShort['confidence'] * sim
                #res = res[['track_id','confidence','metasim']]
        
                baseResShort.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
                #baseRes['confidence'] = baseRes['confidence'] * sim.min()
        
        baseRes = pd.concat([baseResShort,baseRes.tail( len(baseRes)-num )] )
               
        return baseRes.head(self.return_num_preds)
    
    def normpdf(self, x, mean, sd):
        var = float(sd)**2
        pi = 3.1415926
        denom = (2*pi*var)**.5
        num = np.exp(-(x-mean)**2/(2*var))
        return num/denom
    