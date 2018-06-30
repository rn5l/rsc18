'''
Created on 17.04.2018
@author: malte
'''
import numpy as np
import pandas as pd
import time
from sympy.physics.quantum.matrixutils import sparse
import implicit

class ICAGH:
    
    def __init__(self, items=500, min_sim=0.5, normalize=False, sim_weight=False, item_key='track_id', artist_key='artist_id', session_key='playlist_id'):
        self.item_key = item_key
        self.artist_key = artist_key
        self.session_key = session_key
        
        self.items = items
        self.min_sim = min_sim
        self.sim_weight = sim_weight
        self.normalize = normalize
        
    def train(self, data, test=None):
        
        train = data['actions']
        test = test['actions']
        self.artistlist = data['artists']
        self.artistlist.index = self.artistlist['artist_id']
        
        data = pd.concat( [train,test] )
        
        artistids = data[self.artist_key].unique()
        self.n_artists = len(artistids)
        self.artistidmap = pd.Series(data=np.arange(self.n_artists), index=artistids)
        self.artistidmap2 = pd.Series(index=np.arange(self.n_artists), data=artistids)
        self.trackartistmap = data.groupby('track_id').artist_id.min()
        
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.useridmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        tstart = time.time()
        
        data = pd.merge(data, pd.DataFrame({self.artist_key:self.artistidmap.index, 'ArtistIdx':self.artistidmap[self.artistidmap.index].values}), on=self.artist_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:self.useridmap.index, 'SessionIdx':self.useridmap[self.useridmap.index].values}), on=self.session_key, how='inner')
        
        print( 'add index in {}'.format( (time.time() - tstart) ) )
                
        ones = np.ones( len(data) )
        
        row_ind = data.ArtistIdx
        col_ind = data.SessionIdx
        
        self.mat = sparse.csr_matrix((ones, (row_ind, col_ind)))
        self.model = implicit.als.AlternatingLeastSquares( factors=32, iterations=10, regularization=0.01 )
        self.model.fit( self.mat )
        
        self.tmp = self.mat.T.tocsr()
        self.tmp = sparse.csr_matrix((len(col_ind), len(row_ind)))
        
        agh = pd.DataFrame()
        agh['apop'] = train.groupby( [self.artist_key, self.item_key] ).size()
        if self.normalize:
            agh = agh.reset_index()
            agh['apop'] = agh['apop'] / agh.groupby( [self.artist_key] )['apop'].transform('sum')
            #print(agh)
            agh = agh
            #agh = agh.reset_index()
            self.agh = pd.DataFrame()
            self.agh = agh.groupby( [self.artist_key, self.item_key] ).mean()
        else: 
            self.agh = agh
            
        pop = pd.DataFrame()
        pop['pop'] = train.groupby( [self.item_key] ).size()
        pop['pop'] = pop['pop'] / len(train)
        self.pop = pop
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : string
            playlist name
        tracks : int list
            tracks in the current playlist
        playlist_id : int
            playlist identifier
        artists : int list
            artists in the current playlist (per track)
        num_hidden : int
            Number of hidden tracks in the list
            
        Returns
        --------
        res : pandas.DataFrame
            sorted predictions with track_id and confidence
        
        '''   
        
        res = pd.DataFrame()
        
        if playlist_id is not None:
            
            if playlist_id not in self.useridmap.index:
                res['track_id'] = []
                res['confidence'] = []
                return res
            
            #print( self.artistlist['artist_name'][ artists ] )
            
            ca = self.model.recommend( self.useridmap[playlist_id], self.tmp, N=self.items )
            artists = pd.Series( index=self.artistidmap2[ [x[0] for x in ca] ], data=[x[1] for x in ca] )
            artists = artists[ artists >= self.min_sim ]
            
            #print( self.artistlist['artist_name'][ artists.index ] )
            
        if artists is None or len(artists) == 0:
            res['track_id'] = []
            res['confidence'] = []
            return res
        
        idxs = pd.IndexSlice
        sagh = self.agh.loc[ idxs[ artists.index.values,: ] ]
        sagh = sagh.reset_index()
        if self.sim_weight:
            sagh = sagh.merge( artists.to_frame('sim'), left_on='artist_id', right_index=True, how='inner' )
            
        res['confidence'] = sagh.groupby( self.item_key )['apop'].sum()
        if self.sim_weight:
            res['sim'] = sagh.groupby( self.item_key )['sim'].min()
            res['confidence'] *= res['sim']
        res.reset_index(inplace=True)
        res['confidence'] += self.pop['pop'][ res[self.item_key].values ].values
        
        res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
        
        return res.head(500)
    