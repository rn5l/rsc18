'''
Created on 09.04.2018

@author: Iman
'''

import pandas as pd
import json
from helper import inout as IO 
import os

# data folder
SRC_FOLDER = 'data/original/'
TARGET_FOLDER = 'data/online/'

FOLDER_DATA = 'data/data_formatted/'

TEST_FILE = 'challenge_set.json'

#save
PLAYLISTS_FILE = 'playlists.csv'
PLAYLISTS_TRACKS_FILE = 'playlists_tracks.csv'


playlists = {}
playlists['playlist_id'] = []
playlists['name'] = []
playlists['num_tracks'] = []
playlists['num_samples'] = []

playlists_tracks = {}
playlists_tracks['playlist_id'] = []
playlists_tracks['track_id'] = []
playlists_tracks['artist_id'] = []
playlists_tracks['pos'] = []

if __name__ == '__main__':
        
    _, artists, tracks  = IO.load_meta( FOLDER_DATA )
    print(tracks)
    print(artists)
    trackmap = tracks[['track_id','track_uri']]
    trackmap.index = trackmap.track_uri
    artistmap = artists[['artist_id','artist_uri']]
    artistmap.index = artistmap.artist_uri
      
    fp = os.sep.join((SRC_FOLDER, TEST_FILE))
    f = open(fp)
    js = f.read()
    json_str = json.loads(js)
    f.close()
    
    playlists_str = json_str['playlists']
    
    #iterate over playlists
    for row in playlists_str:
        
        playlist_id = row['pid']
        
        playlists['playlist_id'].append( playlist_id )
        if 'name' in row:
            playlists['name'].append( row['name'] )
        else:
            playlists['name'].append( None )
        playlists['num_tracks'].append( row['num_tracks'] )
        playlists['num_samples'].append( row['num_samples'] )
                        
        playlist_tracks = row['tracks']
        
        for track in playlist_tracks:
            
            artist_uri = track['artist_uri']
            artist_id  = artistmap.ix[artist_uri].artist_id
            
            track_uri = track['track_uri']
            track_id  = trackmap.ix[track_uri].track_id
        
            playlists_tracks['playlist_id'].append( playlist_id )
            playlists_tracks['track_id'].append( track_id )
            playlists_tracks['artist_id'].append( artist_id )
            playlists_tracks['pos'].append( track['pos'] )
            
    df_playlists = pd.DataFrame.from_dict(playlists)
    df_playlists_tracks = pd.DataFrame.from_dict(playlists_tracks)
        
    print('saving files...')
    df_playlists.to_csv( TARGET_FOLDER + PLAYLISTS_FILE, index=False)
    df_playlists_tracks.to_csv( TARGET_FOLDER + PLAYLISTS_TRACKS_FILE, index=False)
    
    
