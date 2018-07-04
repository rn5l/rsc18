'''
Created on 29.01.2018

@author: Iman
'''

import spotipy as sp
import os
import pandas as pd
import numpy as np
import time
from spotipy.oauth2 import SpotifyClientCredentials
from _datetime import timedelta
import traceback
import json
import math
from helper import inout

FOLDER_IN = 'data/original/'
FOLDER_OUT = 'data/metadata_combined/'

USER = ''
CLIENT_ID = ''
CLIENT_SECRET = ''
REDIRECT = 'https://localhost:8080/'
BATCH = 50


if __name__ == '__main__':
    
    #initialize spotify api
    ccm = SpotifyClientCredentials( client_id=CLIENT_ID, client_secret=CLIENT_SECRET )
    api = sp.Spotify(client_credentials_manager=ccm)
    
    track_uris = []

    print('LOADING TRACKS ...')
    done = 0
    for file in os.listdir(FOLDER_IN):
        
        filename = os.fsdecode(file)
        if filename.startswith('mpd.slice.'):
            
            data = json.load(open(FOLDER_IN + filename))
            
            for playlist in data["playlists"]:
                for track in playlist["tracks"]:
                    track_uris.append(track["track_uri"].split("spotify:track:")[1])
                    
            done += 1
            if(done%50 == 0):
                print('--DONE FOR {} JSON FILES...'.format(done))
                
    unique_tracks = np.unique(track_uris)
    
    tracks_cnt = np.size(unique_tracks)
    
    print('#TRACKS: ' + str(tracks_cnt) )
    
    metadata = {}
    metadata['track_id'] = []
    metadata['popularity']= []
    metadata['acousticness']= []
    metadata['danceability']= []
    metadata['energy']= []
    metadata['instrumentalness']= []
    metadata['mode']= []
    metadata['loudness']= []
    metadata['speechiness']= []
    metadata['tempo']= []
    metadata['time_signature']= []
    metadata['valence']= []
    
    start = time.time()
         
    print('START CRAWLING SPOTIFY...')
      
    batch_size = BATCH
    
    batches_count = math.ceil(tracks_cnt / batch_size)
    
    print('NO. OF BATCHES TO PROCESS: ' + str(batches_count))
    
    done = 0
    for i in range(0, batches_count):
        batch = [] 
        begin_index = i * batch_size
        end_index = begin_index + batch_size
        
        if (end_index > tracks_cnt):
            end_index = tracks_cnt
        
        batch = unique_tracks[begin_index:end_index:1]
        
        try:
        
            tracks = api.tracks( batch )
              
            for values in tracks.values():
                for track in values:
                    if track != None:
                        metadata['track_id'].append( track['id'] )
                        metadata['popularity'].append( track['popularity'] )
                    else:
                        metadata['track_id'].append( np.nan )
                        metadata['popularity'].append( np.nan )
                    
            features = api.audio_features( batch )
        
            for feature in features: 
                if feature != None:
                    metadata['acousticness'].append( feature['acousticness'] )
                    metadata['danceability'].append( feature['danceability'] )
                    metadata['energy'].append( feature['energy'] )
                    metadata['instrumentalness'].append( feature['instrumentalness'] )
                    metadata['mode'].append( feature['mode'] )
                    metadata['loudness'].append( feature['loudness'] )
                    metadata['speechiness'].append( feature['speechiness'] )
                    metadata['tempo'].append( feature['tempo'] )
                    metadata['time_signature'].append( feature['time_signature'] )
                    metadata['valence'].append( feature['valence'] )
                    
                else:
                    metadata['acousticness'].append( np.nan )
                    metadata['danceability'].append( np.nan )
                    metadata['energy'].append( np.nan )
                    metadata['instrumentalness'].append( np.nan )
                    metadata['mode'].append( np.nan )
                    metadata['loudness'].append( np.nan )
                    metadata['speechiness'].append( np.nan )
                    metadata['tempo'].append( np.nan )
                    metadata['time_signature'].append( np.nan )
                    metadata['valence'].append( np.nan )
            
        except Exception:
            traceback.print_exc()
         
        done = i+1 
        if(done%100==0):
            spent = time.time() - start
            spent_r = str(timedelta(seconds=spent))
            print('-- DONE FOR {} BATCHES! {} MORE BATCHES TO GO. TIME SPENT SO FAR: {}'.format(done, (batches_count-done), spent_r))

    if not os.path.exists(FOLDER_OUT):
        os.makedirs(FOLDER_OUT)
        
    name = FOLDER_OUT + 'all_tracks_metadata.csv'
    inout.ensure_dir( name )      
    print('SAVING THE RESULTS: '+ name)           
    export = pd.DataFrame( metadata )
    export = export[['track_id','popularity','acousticness','danceability','energy','instrumentalness','mode',
                     'loudness','speechiness','tempo','time_signature','valence']]
    export.to_csv(name, sep=',', index=False)
    
    