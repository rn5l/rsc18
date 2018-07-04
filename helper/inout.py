'''
Created on 11.04.2018

@author: malte
'''

import pandas as pd
import numpy as np
import time
import os

# data folder
FOLDER = '/media/mpd-share/data_formated/'
#FOLDER = '/media/mpd-share/data_formated_100k/'
FOLDER = '/media/mpd-share/data_formated_20k/'

#FOLDER = '../data/'
FOLDER_TEST = '/media/mpd-share/sample_random/'
#FOLDER_TEST = '/media/mpd-share/sample_100k_similar/'
#FOLDER_TEST = '/media/mpd-share/sample_100k_random/'
#FOLDER_TEST = '/media/mpd-share/sample_100k_similar/'


# where to save the result file
PLAYLISTS_FILE = "playlists.csv"
TRACKS_FILE = "tracks.csv"
TRACKS_ADD_FILE = "all_tracks_metadata.csv"
ARTISTS_FILE = "artists.csv"
PLAYLISTS_TRACKS_FILE = "playlists_tracks.csv"
PLAYLISTS_TRACKS_FILE_VAL = "playlists_tracks_validation.csv"


def load_dataset( folder_train, folder_test, feather=False ):
    
    print( 'load_dataset' )
    
    train = {}
    test = {}
    
    tstart = time.time()
    
    train['actions'] = load_actions(folder_train, feather)
    playlists, artists, tracks = load_meta(folder_train, feather)
    train['playlists'] = playlists
    train['tracks'] = tracks
    train['artists'] = artists
    
    playlists, actions = load_test(folder_test)
    test['actions'] = actions
    test['playlists'] = playlists
    
    print( ' -- loaded in: {}s'.format( time.time() - tstart ) )
     
    #filter lists
    train['playlists'] = train['playlists'][ ~np.in1d(train['playlists'].playlist_id, test['playlists'].playlist_id.unique()) ]
    train['actions'] = train['actions'][ np.in1d(train['actions'].playlist_id, train['playlists'].playlist_id.unique()) ]
    train['tracks'] = train['tracks'][ np.in1d(train['tracks'].track_id, train['actions'].track_id.unique()) ]
    train['artists'] = train['artists'][ np.in1d(train['artists'].artist_id, train['actions'].artist_id.unique()) ]
    
    print( ' -- filtered in: {}s'.format( time.time() - tstart ) )
        
    tdiff = np.setdiff1d( test['actions'].track_id.unique(), train['tracks'].track_id.unique() )
    
    if len(tdiff) > 0 :
        print( ' -- !!!WARNING!!! cold start items in test {}'.format( tdiff ) )
    
    print(' -- actions: ', len( train['actions'] ))
    print(' -- items: ',train['actions'].track_id.nunique())
    print(' -- lists: ',train['actions'].playlist_id.nunique())

    
    return train, test

def load_actions( folder, feather=False ):
    if feather:
        actions = pd.read_feather( folder + PLAYLISTS_TRACKS_FILE + '.fthr' )
        return actions
    
    actions = pd.read_csv( folder + PLAYLISTS_TRACKS_FILE )
    return actions

def load_actions_hdf5( folder ):
    store = pd.HDFStore( folder + 'store.hdf5' )
    actions = store[PLAYLISTS_TRACKS_FILE]
    return actions

def load_meta( folder, feather=False ):
    if feather:
        playlists = pd.read_feather( folder + PLAYLISTS_FILE + '.fthr' )
        artists = pd.read_feather( folder + ARTISTS_FILE + '.fthr' )
        tracks = pd.read_feather( folder + TRACKS_FILE + '.fthr' )
        return playlists, artists, tracks
    
    playlists = pd.read_csv( folder + PLAYLISTS_FILE )
    artists = pd.read_csv( folder + ARTISTS_FILE )
    tracks = pd.read_csv( folder + TRACKS_FILE )
    return playlists, artists, tracks

def load_meta_hdf5( folder ):
    store = pd.HDFStore( folder + 'store.hdf5' )
    playlists = store[PLAYLISTS_FILE]
    artists = store[ARTISTS_FILE]
    tracks = store[TRACKS_FILE]
    return playlists, artists, tracks

def load_test( folder ):
    lists = pd.read_csv( folder + PLAYLISTS_FILE )
    actions = pd.read_csv( folder + PLAYLISTS_TRACKS_FILE )
    max_pos = actions.groupby('playlist_id').pos.max().to_frame("max_pos")
    lists = lists.merge(max_pos, how='left', left_on='playlist_id', right_index=True )
    lists['in_order'] = lists.max_pos < lists.num_samples.fillna(0)
    lists.sort_values(['num_samples','in_order','playlist_id'], inplace=True)
    
    return lists, actions

def load_meta_track( folder, feather=False ):
    if feather:
        tracks = pd.read_feather( folder + TRACKS_FILE + '.fthr' )
        return tracks
    
    tracks = pd.read_csv( folder + TRACKS_ADD_FILE )
    return tracks

def load_submission( submission ):
    
    pass

def save_submission( folder, frame, file, track='main', team='KAENEN', contact='iman.kamehkhosh@tu-dortmund.de' ):
    
    playlists, artists, tracks = load_meta(folder, feather=True)
        
    fh = open( file, 'w+' )
    
    fh.write( "#SUBMISSION" )
    fh.write( '\n' )
    fh.write( 'team_info,{},{},{}'.format( track, team, contact ) )
    fh.write( '\n' )
    
    frame = frame.merge( tracks[['track_id','track_uri']] )
    frame = frame.sort_values( ['playlist_id','confidence'], ascending=False )
    
    pid = -1
    
    for row in frame.itertuples():
        if row.playlist_id != pid:
            fh.write( '\n' )
            fh.write( str(row.playlist_id) )
            pid = row.playlist_id
        
        fh.write( ',' + row.track_uri )
    
    fh.write('\n')
    fh.close()

def load_validation( folder ):
    if not os.path.isfile( folder + PLAYLISTS_TRACKS_FILE_VAL ):
        return
    lists = pd.read_csv( folder + PLAYLISTS_FILE )
    actions = pd.read_csv( folder + PLAYLISTS_TRACKS_FILE_VAL )
    actions_test = pd.read_csv( folder + PLAYLISTS_TRACKS_FILE )
    max_pos = actions_test.groupby('playlist_id').pos.max().to_frame("max_pos")
    lists = lists.merge(max_pos, how='left', left_on='playlist_id', right_index=True )
    lists['in_order'] = lists.max_pos < lists.num_samples.fillna(0)
    return lists, actions

def convert_hdf5( folder ):
    
    actions = load_actions(folder)
    playlists, artists, tracks = load_meta(folder)
    
    store = pd.HDFStore( folder + 'store.hdf5' )
    store[PLAYLISTS_TRACKS_FILE] = actions
    store[PLAYLISTS_FILE] = playlists
    store[ARTISTS_FILE] = artists
    store[TRACKS_FILE] = tracks
    
def convert_feather( folder ):
    
    actions = load_actions(folder)
    playlists, artists, tracks = load_meta(folder)
    
    actions.to_feather( folder + PLAYLISTS_TRACKS_FILE + '.fthr' )
    playlists.to_feather( folder + PLAYLISTS_FILE + '.fthr' )
    artists.to_feather( folder + ARTISTS_FILE + '.fthr' )
    tracks.to_feather( folder + TRACKS_FILE + '.fthr' )

def ensure_dir(path, file=True):
    if file:
        path = os.path.dirname(path)
    if not os.path.exists( path ):
        os.makedirs( path )

if __name__ == '__main__':
    
    start = time.time()
    #convert_hdf5( FOLDER )
    convert_feather( FOLDER )
    #actions = load_actions_hdf5(FOLDER)
    #load_dataset(FOLDER, FOLDER_TEST, feather=True)
    print( 'loaded in {}s'.format( (time.time() - start) ) )

    