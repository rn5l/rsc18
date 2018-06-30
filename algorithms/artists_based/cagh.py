'''
Created on 07.05.2018
@author: Iman
'''

import os
import pickle
import operator
import pandas as pd

class CAGH(object):
    '''
    classdocs
    '''

    def __init__(self, min_collocation = 5, min_top_track_occurrence=5, folder=None):
        '''
        Constructor
        '''
        self.min_collocation = min_collocation
        self.min_top_track_occurrence = min_top_track_occurrence
        self.folder = folder
        
    def train(self, train, test=None):
        
        self.artists_correlations = {}
        self.artists_tracks_count = {}
        
        self.actions = train['actions']
        
        folder = self.folder
        
        # load artist correlation if already there
        if folder is not None and os.path.isfile( folder + 'artists_correlations.pkl' ):
            self.artists_correlations = pickle.load( open( folder + 'artists_correlations.pkl', 'rb') )
            
        else: #otherwise compute artist correlations and save it
            
            playlists = self.actions.playlist_id.unique()
             
            # for each playlist, save collocated artists
            cnt = 0;
            print('    --building collocations file from %s playlists...' %len(playlists))
            for playlist in playlists:
                
                # artists in the current playlist
                playlist_artists = self.actions[self.actions.playlist_id == playlist].artist_id.unique()
                 
                for a1 in playlist_artists:
                    artist_correlations = {}
                    for a2 in playlist_artists:
                        if a1 != a2:
                            if a2 in artist_correlations:
                                occ = artist_correlations[a2]
                                artist_correlations[a2] = occ+1
                            else:
                                artist_correlations[a2] = 1
                    if a1 in self.artists_correlations:
                        a1_correlated_artists = self.artists_correlations[a1]
                        for a in artist_correlations:
                            if a in a1_correlated_artists:
                                self.artists_correlations[a1][a] +=  artist_correlations[a]
                            else:
                                self.artists_correlations[a1][a] = artist_correlations[a]
                    else:
                        self.artists_correlations[a1] = artist_correlations
             
                cnt +=1
                if cnt%1000 == 0:
                    print("        --done for %s playlists" %cnt)
                    
            # sort the dict for each artist based on the number co-occurrences
            for artist in self.artists_correlations:
                artist_coll_artists = self.artists_correlations[artist]
                sorted_dict = sorted(artist_coll_artists.items(), key=operator.itemgetter(1), reverse=True)
                self.artists_correlations[artist] = sorted_dict
                
            if folder is not None:
                pickle.dump( self.artists_correlations,  open( folder + 'artists_correlations.pkl', 'wb' ) )
            
            if folder is not None and os.path.isfile( folder + 'artists_tracks_count.pkl' ):
                self.artists_tracks_count = pickle.load( open( folder + 'artists_tracks_count.pkl', 'rb') )
            
            else: 
                print('    --find the most popular tracks of each artist for %s artists...' %len(self.actions))
                cnt = 0
                for action in self.actions.itertuples():
                    artist = action.artist_id
                    track = action.track_id
                    
                    track_count = {}
                    
                    if artist in self.artists_tracks_count:
                        if track in self.artists_tracks_count[artist]:
                            count = self.artists_tracks_count[artist][track] 
                            self.artists_tracks_count[artist][track] = count+1
                        else:
                            track_count[track] = 1
                            self.artists_tracks_count[artist].update(track_count)
                    else:
                        track_count[track] = 1
                        self.artists_tracks_count[artist] = track_count
                        
                    cnt +=1
                    if cnt%100000==0:
                        print('        --done for %s actions' %cnt)
                
                # sort the dict for each artist based on the track count
                for artist in self.artists_tracks_count:
                    artist_tracks_count = self.artists_tracks_count[artist]
                    sorted_dict = sorted(artist_tracks_count.items(), key=operator.itemgetter(1), reverse=True)
                    self.artists_tracks_count[artist] = sorted_dict
            
            
                if folder is not None:
                    pickle.dump( self.artists_tracks_count,  open( folder + 'artists_tracks_count.pkl', 'wb' ) )

    def predict( self, plname, actions, playlist_id=None, artists=None ):
        
        res_dict = {}
        res_dict['track_id'] = []
        res_dict['confidence'] = []
        
        if actions is not None and len(actions) > 0:
            for artist in artists:
                # first top tracks of the current artist
                artist_top_tracks = self.artists_tracks_count[artist]
                
                for tt in artist_top_tracks:
                    if tt[1] >= self.min_top_track_occurrence and tt[0] not in actions:
                        res_dict['track_id'].append( tt[0] )
                        res_dict['confidence'].append( tt[1] )
                    else:
                        break
                
                                
                # then top tracks of the collocated artists
                collocated_artists = self.artists_correlations[artist]
                if collocated_artists is not None:
                    for col_art in collocated_artists:
                        if  col_art[1] >= self.min_collocation:
                        
                            col_art_top_tracks = self.artists_tracks_count[col_art[0]]
                            
                            for tt in col_art_top_tracks:
                                if tt[1] >= self.min_top_track_occurrence and tt[0] not in actions:
                                    res_dict['track_id'].append( tt[0] )
                                    res_dict['confidence'].append( tt[1] )
                                else:
                                    break
                        else:
                            break
                
        res = pd.DataFrame.from_dict(res_dict)
        res.sort_values(by='confidence', ascending=False, inplace=True)
        
        return res.head(500)
        
        