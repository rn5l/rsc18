'''
Created on Jun 11, 2018

@author: malte
'''
from helper import inout
import numpy as np
import pandas as pd
import math
from helper import eval as vl

class EpochEval(object):
    '''
    classdocs
    '''

    def __init__(self, folder, num_lists=100):
        '''
        Constructor
        '''
    
        vlists, truth = inout.load_validation(folder)
        tlists, test = inout.load_test(folder)
                              
        tlists.sort_values('num_samples', inplace=True)
        lists = tlists.reset_index(drop=True)
                
        if num_lists < len(tlists):
            orgcat = math.ceil(len(tlists) / 10)
            percat = int(num_lists / 10)
            indices = list( range(percat) )
            for i in range(1, 10):
                indices += list( range(orgcat * i, orgcat * i + percat) )
            tlists = tlists.ix[indices]
        
        truth = truth[truth.playlist_id.isin( tlists.playlist_id.unique() )]
        test = test[test.playlist_id.isin( tlists.playlist_id.unique() )]
                 
        self.lists = tlists
        self.test = test
        self.truth = truth
        self.folder = folder
            
    def callback(self, algo):
        
        results = {}

        results['playlist_id'] = []
        results['track_id'] = []
        results['confidence'] = []
        
        self.truth.sort_values('playlist_id', inplace=True)

        plidmap = pd.Series(index=list(self.test.playlist_id.unique()), data=range(len(self.test.playlist_id.unique())))
        start = np.r_[ 0, self.test.groupby('playlist_id').size().cumsum().values ]
        tracks = self.test.track_id.values
        artists = self.test.artist_id.values
    
        done = 0

        for row in list(zip(self.lists['playlist_id'], self.lists['name'])):
        
            pid, name = row
        
            if pid in plidmap:
                sidx = plidmap[pid]
                s = start[sidx]
                e = start[sidx + 1]
                actions = tracks[s:e]
                artist_ids = artists[s:e]
            else:
                actions = None
                artist_ids = None
        
            res = algo.predict(name, actions, playlist_id=pid, artists=artist_ids)
        
            results['playlist_id'] += [pid] * len(res)
            results['track_id'] += list(res.track_id.values)
            results['confidence'] += list(res.confidence.values)
        
            done += 1
    
        results = pd.DataFrame.from_dict( results )
        
        eval_res, _ = vl.evaluate(results, self.folder, strict=False, preloaded=(self.lists,self.truth))
        
        print( eval_res )
    
                
