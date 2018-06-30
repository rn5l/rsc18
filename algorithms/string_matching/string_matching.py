'''
Created on 09.05.2018

@author: malte
'''

import numpy as np
import pandas as pd
import math
import time
import random

from nltk.metrics import edit_distance
from nltk import stem, tokenize as tokenise
from nltk.corpus import wordnet as wn
from fuzzywuzzy import fuzz
from unittest.mock import inplace
import re

class StringMatching(object):
    '''
    classdocs
    '''
    
    def __init__(self, weight_gpop=0, stemm=True, stemmer='porter', tokenize=True, clean=True, synonyms=False, fuzzy=True, fuzz_thres=0, add_artists=False, add_albums=False, return_num_predictions=500):
        '''
        Constructor
        '''
        self.weight_gpop = weight_gpop
        self.return_num_predictions = return_num_predictions
        self.add_artists = add_artists
        self.add_albums = add_albums
        
        self.stemm = stemm
        self.tokenize = tokenize
        self.clean = clean 
        self.fuzzy = fuzzy
        self.fuzz_thres = fuzz_thres
        
        if stemmer == 'wn':
            self.stemmer = stem.WordNetLemmatizer()
        elif stemmer == 'porter':
            self.stemmer = stem.PorterStemmer()
        elif stemmer == 'snowball':
            self.stemmer = stem.SnowballStemmer('english')
        self.stemmers = stemmer
         
        self.synonyms = synonyms
        
    def init(self, train, test):
        pass
    
    def train(self, train, test=None):
        
        tstart = time.time()
        
        self.playlists = train['playlists']
        self.actions = train['actions']
        
        if self.add_artists:
            new_actions = pd.DataFrame()
            new_actions['count']= self.actions.groupby(['artist_id','track_id']).size()
            new_actions = new_actions.reset_index()
            max_pl = self.playlists.playlist_id.max()
            new_actions['playlist_id'] = new_actions.artist_id.transform( lambda x: max_pl + x )
            self.actions = pd.concat( [ self.actions, new_actions ] )
            
            new_lists = pd.DataFrame()
            new_lists['artist_id'] = new_actions.groupby( ['playlist_id'] ).artist_id.min()
            new_lists = new_lists.reset_index()
            new_lists = new_lists.merge( train['artists'][ ['artist_id', 'artist_name'] ], on='artist_id', how='inner' )
            new_lists['name'] = new_lists['artist_name']
            del new_lists['artist_name']
            self.playlists = pd.concat( [ self.playlists, new_lists ] )
        
        if self.add_albums:
            train['tracks']['album_id'] = train['tracks']['album_uri'].astype('category').cat.codes
            self.actions = self.actions.merge( train['tracks'][['track_id','album_id']], on='track_id', how='inner' )
            new_actions = pd.DataFrame()
            new_actions['count']= self.actions.groupby(['album_id','track_id']).size()
            new_actions = new_actions.reset_index()
            max_pl = self.playlists.playlist_id.max()
            new_actions['playlist_id'] = new_actions.album_id.transform( lambda x: max_pl + x )
            self.actions = pd.concat( [ self.actions, new_actions ] )
            
            new_lists = pd.DataFrame()
            new_lists['album_id'] = new_actions.groupby( ['playlist_id'] ).album_id.min()
            new_lists = new_lists.reset_index()
            new_lists = new_lists.merge( train['tracks'][ ['album_id', 'album_name'] ], on='album_id', how='inner' )
            new_lists['name'] = new_lists['album_name']
            del new_lists['album_name']
            self.playlists = pd.concat( [ self.playlists, new_lists ] )
                
        #normalize playlist names
        self.playlists['name'] = self.playlists['name'].apply(lambda x: self.normalize(str(x), self.tokenize, self.stemm, self.clean))        
        self.playlists['name_id'] = self.playlists['name'].astype( 'category' ).cat.codes
        self.playlists['count'] = self.playlists.groupby( ['name_id'] ).name_id.transform( 'count' )
        
        self.nameidmap = pd.Series( index=self.playlists['name'], data=self.playlists['name_id'].values )
        self.nameidmap.drop_duplicates(inplace=True)
        
        self.actions = self.actions.merge( self.playlists[['playlist_id', 'name_id']], on='playlist_id', how='inner' )
        self.actions.sort_values( ['name_id','playlist_id','pos'], inplace=True ) 
        self.actions['tmp'] = range(len(self.actions))
        
        del self.actions['tmp']
        
        pop = pd.DataFrame()
        pop['popularity'] = train['actions'].groupby( 'track_id' ).size()
        pop.reset_index(inplace=True)
        pop['confidence'] = pop['popularity'] / len(train['actions'])
        pop.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
        self.pop = pop[['track_id','confidence']]
        
        self.pop.index = self.pop['track_id']
        
        self.not_enough = 0
        
        print( ' -- finished training in {}s'.format( (time.time() - tstart) ) )
        
    def predict(self, name=None, tracks=None, artists=None, playlist_id=None, num_hidden=None):
        
        res = pd.DataFrame()
        tracks = tracks if tracks is not None else []
        
        if name is None or type(name) is float:
            res_dict = {}
            res_dict['track_id'] = []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)
            
        name = self.normalize(str(name), self.tokenize, self.stemm, self.clean)
        
        if not name in self.nameidmap and self.fuzzy:
        
            self.playlists['match'] = self.playlists['name'].apply( lambda n: fuzz.ratio(n,name) )
            self.playlists.sort_values( ['match','count','num_followers'], ascending=False, inplace=True )
            
            if self.playlists['match'].values[0] >= self.fuzz_thres: 
                new_name = self.playlists['name'].values[0]
                #print( name + ' => ' + new_name )
                #print( self.playlists.head(10) )
                name = new_name
            else:
                new_name = self.playlists['name'].values[0]
                #print( 'not good enough: ' + name + ' => ' + new_name )
                 
        
        syn = []
        if self.synonyms:
            #for part in name.split( ' ' ):
            if len(name.split( ' ' )) == 1:
                syns = wn.synsets(name)
                print( name )
                if len(syns) > 0:
                    syns[0].lemmas()
                    tmp = [ lemma.name() for lemma in syns[0].lemmas() ]
                    if name in tmp:
                        tmp.remove( name )
                    print(tmp)
                    syn.extend( tmp )
        
        if name in self.nameidmap:
            name_id = self.nameidmap[name]
                        
            #actions_for_name = self.actions.ix[start:end]
            actions_for_name = self.actions.ix[ self.actions.name_id == name_id ]
             
            res['confidence'] = actions_for_name.groupby( 'track_id' ).size()
            res.reset_index(inplace=True)
            res['confidence'] += self.pop.confidence[ res.track_id.values ].values
            res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
            
        else:
                        
            res['track_id'] = []
            res['confidence'] = []
        
        for word in syn:
            
            word = self.normalize(word, self.tokenize, self.stemm)
            
            if word in self.nameidmap:
                
                name_id = self.nameidmap[word]
                
                actions_for_name = self.actions[ self.actions.name_id == name_id ]
                 
                res_syn = pd.DataFrame() 
                
                res_syn['tmp'] = actions_for_name.groupby( 'track_id' ).size()
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
        
        if len(res) < self.return_num_predictions:
            self.not_enough += 1
        
        res = res[~np.in1d( res.track_id, tracks )]
        
        return res.head( self.return_num_predictions )
    
        
    def normalize(self, s, tokenize=True, stemm=True, clean=False):
        if clean:
            s = re.sub( r"[.,\/#!$§\^\*;:{}=\_´`~()@]", ' ', s )
            s = re.sub( r"\s+", ' ', s )
        
        if tokenize:
            words = tokenise.wordpunct_tokenize(s.lower().strip())
        else:
            words = s.lower().strip().split( ' ' )
        if stemm:
            if self.stemmers != 'wn':
                return ' '.join([self.stemmer.stem(w) for w in words])
            else:
                return ' '.join([self.stemmer.lemmatize(w) for w in words])
        else:
            return ' '.join([w for w in words])
 
    def fuzzy_match(self, s1, s2):
        return edit_distance(s1, s2)
