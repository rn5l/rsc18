'''
Created on 17.04.2018

@author: malte
'''

import implicit
from nltk import stem as stem, tokenize as tokenise

from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from scipy import sparse

class ImplicitStringMatch:
    
    def __init__(self, factors=32, neighbors=20, fuzzy=True, use_count=False, normalize=False, sim_weight=True, add_artists=False, item_key='track_id', artist_key='artist_id', session_key='playlist_id'):
        self.item_key = item_key
        self.artist_key = artist_key
        self.session_key = session_key
        
        self.factors = factors
        self.use_count = use_count
        self.add_artists = add_artists
        
        self.fuzzy = fuzzy
        
        self.neighbors = neighbors
        self.sim_weight = sim_weight
        self.normalize = normalize
        
        self.stemmer = stem.PorterStemmer()
        
    def train(self, train, test=None):
        
        self.actions = train['actions']
        self.playlists = train['playlists']
        #datat = test['actions']
        
        if self.add_artists:
            new_actions = pd.DataFrame()
            new_actions['count']= self.actions.groupby(['artist_id','track_id']).size()
            new_actions = new_actions.reset_index()
            max_pl = self.playlists.playlist_id.max()
            new_actions['playlist_id'] = new_actions.artist_id.transform( lambda x: max_pl + x )
            self.actions = pd.concat( [ self.actions, new_actions ], sort=False )
            
            new_lists = pd.DataFrame()
            new_lists['artist_id'] = new_actions.groupby( ['playlist_id'] ).artist_id.min()
            new_lists = new_lists.reset_index()
            new_lists = new_lists.merge( train['artists'][ ['artist_id', 'artist_name'] ], on='artist_id', how='inner' )
            new_lists['name'] = new_lists['artist_name']
            del new_lists['artist_name']
            self.playlists = pd.concat( [ self.playlists, new_lists ], sort=False )
        
        #normalize playlist names
        self.playlists['name'] = self.playlists['name'].apply(lambda x: self.normalise(str(x), True, True))
        self.playlists['name_id'] = self.playlists['name'].astype( 'category' ).cat.codes
        self.playlists['count'] = self.playlists.groupby('name_id')['name_id'].transform('count')
        
        self.nameidmap = pd.Series( index=self.playlists['name'], data=self.playlists['name_id'].values )
        self.nameidmap.drop_duplicates(inplace=True)
        self.nameidmap2 = pd.Series( index=self.playlists['name_id'], data=self.playlists['name'].values )
        self.nameidmap2.drop_duplicates(inplace=True)
        
        self.actions = self.actions.merge( self.playlists[['playlist_id', 'name_id']], on='playlist_id', how='inner' )
                        
        pop = pd.DataFrame()
        pop['popularity'] = train['actions'].groupby( 'track_id' ).size()
        pop.reset_index(inplace=True)
        pop['confidence'] = pop['popularity'] / len( train['actions'] )
        pop.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
        self.pop = pop[['track_id','confidence']]
        
        self.pop.index = self.pop['track_id']
        
        #MF PART
        itemids = self.actions[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        self.itemidmap2 = pd.Series(index=np.arange(self.n_items), data=itemids)
         
        self.actions = pd.merge(self.actions, pd.DataFrame({self.item_key:self.itemidmap.index, 'ItemIdx':self.itemidmap[self.itemidmap.index].values}), on=self.item_key, how='inner')
         
        datac = pd.DataFrame()
        datac['count'] = self.actions.groupby( ['name_id','ItemIdx'] ).size()
        datac.reset_index( inplace=True )
        data = datac
                 
        if self.use_count:
            datam = data['count']
        else:
            datam = np.ones( len(data) )
             
        #row_ind = data.ItemIdx
        #col_ind = data.name_id
        
        col_ind = data.ItemIdx
        row_ind = data.name_id
         
        self.mat = sparse.csr_matrix((datam, (row_ind, col_ind)))
         
        self.model = implicit.als.AlternatingLeastSquares( factors=self.factors, iterations=10, regularization=0.07, use_gpu=False )
        #self.model = implicitu.bpr.BaysianPersonalizedRanking( factors=self.factors, iterations=self.epochs )
        self.model.fit(self.mat)
        
        self.tmp = self.mat.T
        #self.tmp = sparse.csr_matrix( ( len(col_ind.unique()), len(row_ind.unique()) ) )
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        tracks = [] if tracks is None else tracks
        
        res = pd.DataFrame()
        
        if name is None or type(name) is float:
            res_dict = {}
            res_dict['track_id'] = []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)
        
        name = self.normalise(str(name), True, True)
        
        if not name in self.nameidmap:
        
            self.playlists['match'] = self.playlists['name'].apply( lambda n: fuzz.ratio(n,name) )
            self.playlists.sort_values( ['match','count','num_followers'], ascending=False, inplace=True )
                        
            if self.playlists['match'].values[0] > 60:
#                 playlists = playlists.head(10)
#                 playlists['num'] = playlists.groupby('name')['name'].transform('count')
#                 playlists.sort_values( 'num', ascending=False, inplace=True )
                
                new_name = self.playlists['name'].values[0]
                
                #print( name + ' => ' + new_name )
                #print( playlists )
            
                name = new_name
                
        #print( 'imatch' )
        #print( '    name: ' + name )
        
        if name in self.nameidmap:
            
            name_id = self.nameidmap[name]
            
            actions_for_name = self.actions[ self.actions.name_id == name_id ]
            
            res['confidence'] = actions_for_name.groupby( 'track_id' ).size()
            res.reset_index(inplace=True)
            res['confidence'] += self.pop.confidence[ res.track_id.values ].values
            res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
            
            if self.neighbors > 0:
            
                similar = self.model.similar_items(name_id, N=self.neighbors)
                similar = pd.DataFrame({'name_id':[x[0] for x in similar], 'conf':[x[1] for x in similar]})
                
                actions_all = self.actions[ np.in1d( self.actions.name_id, similar.name_id.values ) ]
                actions_all = actions_all.merge( similar, on='name_id', how='inner' )
                          
                res_syn = pd.DataFrame() 
                
                if self.sim_weight:
                    res_syn['tmp'] = actions_all.groupby( ['track_id'] ).conf.sum()
                else:
                    res_syn['tmp'] = actions_all.groupby( ['track_id'] ).size()
                    
                res_syn.reset_index(inplace=True)
                res_syn['tmp'] += self.pop.confidence[ res_syn.track_id.values ].values
                
                if len(res) > 0:
                    res = res.merge( res_syn, how="left", on='track_id' )
                    res['confidence'] += res['tmp'].fillna(0)
                    del res['tmp']
               
                res_syn['confidence'] = res_syn['tmp']
                del res_syn['tmp']
                
                mask = ~np.in1d( res_syn.track_id, res['track_id'] )
                if mask.sum() > 0:
                    res = pd.concat( [ res, res_syn[mask] ] )
            
        else:
            
            res['track_id'] = []
            res['confidence'] = []
        
        res = res[~np.in1d( res.track_id, tracks )]
        res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
                
        return res.head(500)
    
    def normalise(self, s, tokenize=True, stemm=True):
        if tokenize:
            words = tokenise.wordpunct_tokenize(s.lower().strip())
        else:
            words = s.lower().strip().split( ' ' )
        if stemm:
            return ' '.join([self.stemmer.stem(w) for w in words])
        else:
            return ' '.join([w for w in words])
        
    
