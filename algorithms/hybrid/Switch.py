'''
Created on 17.04.2018
@author: malte
'''

class Switch:
    
    def __init__(self, algs, max_samples):
        self.algs = algs
        self.max_samples = max_samples
        
    def train(self, data, test=None):
        
        self.lists = test['playlists']
        
        for a in self.algs:
            a.train(data, test=test)
        
    def predict(self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None):
        
        num_samples = self.lists[self.lists.playlist_id == playlist_id].num_samples.min()
        
        i = 0
        for threshold in self.max_samples:
            if num_samples < threshold:
                break
            else:
                i += 1
        
        baseRes = self.algs[i].predict( name, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden )
        
        if len(baseRes) > baseRes.track_id.nunique():
            raise Exception( 'no unique recs ', type(self.base).__name__ )
            
        return baseRes
    