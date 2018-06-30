'''
Created on 09.04.2018

@author: Iman
'''

import pandas as pd
import os
import json

# data folder
FOLDER = "/media/mpd-share/data/"

# where to save the result file
PLAYLISTS_FILE = "../data/playlists.csv"
TRACKS_FILE = "../data/tracks.csv"
PLAYLISTS_TRACKS_FILE = "../data/playlists_tracks.csv"

# data dictionaries
playlists = {}
all_tracks = {}
playlists_tracks = {}

if __name__ == '__main__':
    
    results = pd.DataFrame([])
    
    filenames = os.listdir(FOLDER)
    
    first_track = True
    for i in range(0, len(filenames)):
#   for filename in filenames:
        if filenames[i].startswith("mpd.slice.") and filenames[i].endswith(".json"):
            fp = os.sep.join((FOLDER, filenames[i]))
            josn_str = json.load(open(fp))
            playlists_str = josn_str['playlists']
            
            # load json file in a dataframe
            df_file_playlists = pd.DataFrame.from_dict(playlists_str)
            
#             if i == 0:
#                 df_playlists = df_file_playlists
#             else:
#                 df_playlists = df_playlists.append(df_file_playlists, ignore_index=True)
            
            #iterate over playlists
            for index, row in df_file_playlists.iterrows():
                pid = row['pid']
                
                playlist_info = {'name':'','num_tracks':'', 'num_artists':'',
                'num_albums':'','num_followers':'','num_edits':'',
                'duration_ms':'','modified_at':'','collaborative':'','description':''}
    
                playlist_info['name'] = row['name']
                playlist_info['num_tracks'] = row['num_tracks']
                playlist_info['num_artists'] = row['num_artists']
                playlist_info['num_albums'] = row['num_albums']
                playlist_info['num_followers'] = row['num_followers']
                playlist_info['num_edits'] = row['num_edits']
                playlist_info['duration_ms'] = row['duration_ms']
                playlist_info['modified_at'] = row['modified_at']
                playlist_info['collaborative'] = row['collaborative']
                playlist_info['description'] = row['description']
                
                playlists[pid] = playlist_info
                
                playlist_tracks = row['tracks']
                
                playlists_tracks_line = ""
                for track in playlist_tracks:
                    track_uri = track['track_uri']
                    pos = track['pos']
                    playlists_tracks_line += str(pos) + ":" + track_uri + ";"
                    
                    track_info = {'track_name':'', 'artist_uri':'',
                              'artist_name':'','album_uri':'','duration_ms':'','album_name':''}
                    
                    if track_uri not in all_tracks:
                        track_info['track_name'] = track['track_name']
                        track_info['artist_uri'] = track['artist_uri']
                        track_info['artist_name'] = track['artist_name']
                        track_info['album_uri'] = track['album_uri']
                        track_info['duration_ms'] = track['duration_ms']
                        track_info['album_name'] = track['album_name']
                        
                        all_tracks[track_uri]  = track_info
                
                
                playlists_tracks[pid] = playlists_tracks_line[:-1]
        
        if (i%10==0):
            print('done for', i, 'files from 1000')
            
    df_playlists = pd.DataFrame.from_dict(playlists,orient="index")
    df_tracks = pd.DataFrame.from_dict(all_tracks,orient="index")
    df_playlists_tracks = pd.DataFrame.from_dict(playlists_tracks, orient="index")
    
#     df_playlists = df_playlists.drop("tracks", axis=1)
    
    print('saving files...')
    df_playlists.to_csv(PLAYLISTS_FILE, ";", index=True)
    df_tracks.to_csv(TRACKS_FILE, ";", index=True)
    df_playlists_tracks.to_csv(PLAYLISTS_TRACKS_FILE, ";", index=True, header=False)
    
    
