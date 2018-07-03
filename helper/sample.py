'''
Created on 11.04.2018

@author: malte
'''

import pandas as pd
import numpy as np
import time
from helper import inout
import math

# data folder
FOLDER = '/media/mpd-share/data_formated_100k/'
FOLDER_TEST = '/media/mpd-share/online/'
FOLDER_TARGET = '/media/mpd-share/sample4_100k'

FOLDER_TARGET_TRAIN = '/media/mpd-share/data_formated'


def create_similar_sample( playlists, actions, challenge_set, target, reduce=1 ):
    print( 'create similar sample' )
    tstart = time.time()
    
    #only lists with support > 2
    actions['sup'] = actions.groupby( 'track_id' )['track_id'].transform( 'size' )
    good = actions.groupby('playlist_id')['sup'].min()
    good = good[good > 2]
    actions = actions[ np.in1d( actions.playlist_id, good.index ) ]
    del actions['sup']
    playlists = playlists[ np.in1d( playlists.playlist_id, good.index ) ]
    
    print( ' -- removed support <= 2' )
    
    sample = {}
    sample['name'] = []
    sample['num_samples'] = []
    sample['num_tracks'] = []
    sample['playlist_id'] = []
    playlist_set = set()
    
    test_set = pd.DataFrame()
    
    cpls = challenge_set[0]
    cactions = challenge_set[1]
    
    itemerror = 0
    doubleerror = 0
    
    if reduce < 1:
        index = []
        #create smaller test set 
        for i in range(10): #10 tasks with 1000 playlist
            start = i * 1000
            end = start + math.floor( 1000*reduce )
            index += list(range(start,end))
    
        cpls = cpls.ix[index]
   
    i = 0
    for list_example in cpls.itertuples():
        
        good = False 
        while not good:
            
            slist = playlists[ ( playlists.num_tracks > list_example.num_tracks-10 ) & ( playlists.num_tracks < list_example.num_tracks + 10 ) & ( playlists.num_tracks > list_example.num_samples ) ].sample( 1 )
            pid = slist.playlist_id.values[0]
            
            if pid in playlist_set:
                doubleerror += 1
                print( 'double error pid ', pid )
                continue
                            
            #reproduce from random playlist
            eactions = cactions[ cactions.playlist_id == list_example.playlist_id ]
                    
            if list_example.num_samples > 0 :
                
                lactions = actions[ actions.playlist_id == pid ]
                
                if eactions.pos.max() >= list_example.num_samples: #random tracks
                    tactions = lactions.sample( list_example.num_samples )
                else: #first tracks
                    tactions = lactions.head( list_example.num_samples )
                
                if len( tactions ) is 0:
                    print( 'no actions received...' )
                    print( pid )
                    print( list_example.num_samples )
                    print( list_example.num_tracks )
                    print( slist.num_tracks )
                    exit()
                   
                test_set = pd.concat([test_set, tactions])
                
            sample['num_samples'].append( list_example.num_samples )
            sample['num_tracks'].append( slist.num_tracks.values[0] )
            sample['playlist_id'].append( pid )
            playlist_set.add(pid)
            
            #print( list_example.name )
            if list_example.name is not None and list_example.name != '' and list_example.name != 'nan' and not type(list_example.name) is float:
                sample['name'].append( slist.name.values[0] )
                #print( 'added' )
            else:
                sample['name'].append( None )
                
            good = True
            
        i += 1
        
        if i % 100 == 0:
            print( ' -- processed ',i,' example lists ', diff(tstart) )
    
    print( ' -- double lists: ', doubleerror )
    print( ' -- item support 1: ', itemerror )
    print( ' -- create frames', diff(tstart) )
    
    sample = pd.DataFrame.from_dict(sample)
    
    ids = sample.playlist_id.unique()
    validation_set = actions[ np.in1d( actions.playlist_id, ids ) ]
    
    print( ' -- remove test from validation', diff(tstart) )
    
    validation_set.to_csv( target + 'playlists_tracks_full.csv', index=False )
    
    #remove training tracks from validation
    test_set['test'] = 1
    validation_set = validation_set.merge( test_set[['playlist_id','track_id','pos','test']], on=['playlist_id','track_id','pos'], how='outer' )
    validation_set = validation_set[validation_set.test != 1]
    
    del validation_set['test']
    del test_set['test']
    
    print( ' -- save results', diff(tstart) )
    
    sample.to_csv( target + 'playlists.csv', index=False )
    test_set.to_csv( target + 'playlists_tracks.csv', index=False )
    validation_set.to_csv( target + 'playlists_tracks_validation.csv', index=False )
    
    print( ' -- finished', diff(tstart) )

def create_random_sample( playlists, actions, challenge_set, target, reduce=1 ):
    
    print( 'create random sample' )
    tstart = time.time()
    
    #only lists with support > 2
    actions['sup'] = actions.groupby( 'track_id' )['track_id'].transform( 'size' )
    good = actions.groupby('playlist_id')['sup'].min()
    good = good[good >= 2]
    actions = actions[ np.in1d( actions.playlist_id, good.index ) ]
    del actions['sup']
    playlists = playlists[ np.in1d( playlists.playlist_id, good.index ) ]
    
    print( ' -- removed support <= 2' )
    
    sample = {}
    sample['name'] = []
    sample['num_samples'] = []
    sample['num_tracks'] = []
    sample['playlist_id'] = []
    playlist_set = set()
    
    test_set = pd.DataFrame()
    
    cpls = challenge_set[0]
    cactions = challenge_set[1]
    
    itemerror = 0
    doubleerror = 0
    
    if reduce < 1:
        index = []
        #create smaller test set 
        for i in range(10): #10 tasks with 1000 playlist
            start = i * 1000
            end = start + math.floor( 1000*reduce )
            index += list(range(start,end))
    
        cpls = cpls.ix[index]
   
    i = 0
    for list_example in cpls.itertuples():
        
        good = False 
        while not good:
            
            slist = playlists[ playlists.num_tracks > list_example.num_samples ].sample( 1 )
            pid = slist.playlist_id.values[0]
            
            if pid in playlist_set:
                doubleerror += 1
                print( 'double error pid ', pid )
                continue
                            
            #reproduce from random playlist
            eactions = cactions[ cactions.playlist_id == list_example.playlist_id ]
                    
            if list_example.num_samples > 0 :
                
                lactions = actions[ actions.playlist_id == pid ]
                
                if eactions.pos.max() >= list_example.num_samples: #random tracks
                    tactions = lactions.sample( list_example.num_samples )
                else: #first tracks
                    tactions = lactions.head( list_example.num_samples )
                
                if len( tactions ) is 0:
                    print( 'no actions received...' )
                    print( pid )
                    print( list_example.num_samples )
                    print( list_example.num_tracks )
                    print( slist.num_tracks )
                    exit()
                   
                test_set = pd.concat([test_set, tactions])
                
            sample['num_samples'].append( list_example.num_samples )
            sample['num_tracks'].append( slist.num_tracks.values[0] )
            sample['playlist_id'].append( pid )
            playlist_set.add(pid)
            
            if list_example.name is not None and list_example.name != '' and list_example.name != 'nan':
                sample['name'].append( slist.name.values[0] )
            else:
                sample['name'].append( None )
                
            good = True
            
        i += 1
        
        if i % 100 == 0:
            print( ' -- processed ',i,' example lists ', diff(tstart) )
    
    print( ' -- double lists: ', doubleerror )
    print( ' -- create frames', diff(tstart) )
    
    sample = pd.DataFrame.from_dict(sample)
    
    ids = sample.playlist_id.unique()
    validation_set = actions[ np.in1d( actions.playlist_id, ids ) ]
    
    print( ' -- remove test from validation', diff(tstart) )
    
    validation_set.to_csv( target + 'playlists_tracks_full.csv', index=False )
    
    #remove training tracks from validation
    test_set['test'] = 1
    validation_set = validation_set.merge( test_set[['playlist_id','track_id','pos','test']], on=['playlist_id','track_id','pos'], how='outer' )
    validation_set = validation_set[validation_set.test != 1]
    
    del validation_set['test']
    del test_set['test']
    
    print( ' -- save results', diff(tstart) )
    
    sample.to_csv( target + 'playlists.csv', index=False )
    test_set.to_csv( target + 'playlists_tracks.csv', index=False )
    validation_set.to_csv( target + 'playlists_tracks_validation.csv', index=False )
    
    print( ' -- finished', diff(tstart) )
    
def check_sample(  ):
    pass

def create_random_training_sample( playlists, artists, tracks, actions, target, reduce=0.1 ):
    
    playlists = playlists.sample( math.floor( len( playlists )*reduce ) )
    actions = actions[ np.in1d( actions.playlist_id, playlists.playlist_id ) ]
    artists = artists[ np.in1d( artists.artist_id, actions.artist_id.unique() ) ]
    tracks = tracks[ np.in1d( tracks.track_id, actions.track_id.unique() ) ]

    playlists.to_csv( target + 'playlists.csv', index=False )
    tracks.to_csv( target + 'tracks.csv', index=False )
    artists.to_csv( target + 'artists.csv', index=False )
    actions.to_csv( target + 'playlists_tracks.csv', index=False )
    

def diff( start ):
    return (time.time() - start)

if __name__ == '__main__':
    
    actions = inout.load_actions(FOLDER, feather=True)
    playlists, artists, tracks = inout.load_meta(FOLDER, feather=True)
    
    challenge_set = inout.load_test(FOLDER_TEST)
    
    #create_random_training_sample(playlists, artists, tracks, actions, FOLDER_TARGET_TRAIN + '/', reduce=0.02)
    
    create_random_sample( playlists, actions, challenge_set, FOLDER_TARGET+'_random/', reduce=0.1 )
    create_similar_sample( playlists, actions, challenge_set, FOLDER_TARGET+'_similar/', reduce=0.1 )
    
    
