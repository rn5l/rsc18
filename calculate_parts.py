'''
Created on 13.04.2018

@author: malte
'''
import time

from algorithms.baseline.Popular import MostPopular
from algorithms.baseline.SolutionScaled import Solution
from algorithms.hybrid.Fill import Fill
from algorithms.hybrid.Switch import Switch
from algorithms.hybrid.Weighted import Weighted
from algorithms.knn.iknn import ItemKNN
from algorithms.knn.sknn import SessionKNN
from algorithms.knn_disk.knn_disk import KNNDisk
from algorithms.mf.implicitu import Implicit
from algorithms.mf.implicitc import ColdImplicit
from algorithms.string_matching.implicit_match import ImplicitStringMatch
from algorithms.string_matching.string_matching import StringMatching
from helper import inout
from helper.eval import evaluate
import numpy as np
import pandas as pd
import gc

NUM_RECOMMENDATIONS=500

# data folder
FOLDER_TRAIN = 'data/data_formatted_50k/'
FOLDER_TEST = 'data/sample_50k_similar/'

def main():
    
    algs = {}
    
    #calculate the solutions for each single approach
    
    mp = MostPopular()
    
    algs['diskknn'] = Fill( KNNDisk( 1000, tf_method='ratio-s50', idf_method='log10', similarity='cosine', sim_denom_add=0, folder=FOLDER_TEST ), mp )
    algs['sknn'] = Fill( SessionKNN( 2000, 0, idf_weight=1, folder=FOLDER_TEST ), mp )
    algs['iknn'] = Fill( ItemKNN( 100, alpha=0.75, idf_weight=1, folder=FOLDER_TEST ), mp )
    algs['implicit'] = Fill( ColdImplicit( 300, epochs=10, reg=0.08, idf_weight=1, algo='als' ), mp )
    #algs['implicit'] = Fill( Implicit( 300, epochs=10, reg=0.08, algo='als' ), mp )

    algs['smatch'] = Fill( StringMatching(), mp )
    algs['imatch'] = Fill( ImplicitStringMatch( 128, add_artists=True ), mp )
    
    #start processing
    
    train, test = inout.load_dataset( FOLDER_TRAIN, FOLDER_TEST, feather=True )
    
    results = {}
    results_time = {}
    
    for k, v in algs.items():
        
        tstart = time.time()
        v.train( train, test=test )
        print( 'trained {} in {}s'.format( k, (time.time() - tstart) ) )
    
    
        results[k] = {}
        results[k]['playlist_id'] = []
        results[k]['track_id'] = []
        results[k]['confidence'] = []
        results_time[k] = 0
            
        test['actions'].sort_values( 'playlist_id', inplace=True )
        plidmap = pd.Series( index=list(test['actions'].playlist_id.unique()), data=range(len(test['actions'].playlist_id.unique())) )
        start = np.r_[ 0, test['actions'].groupby('playlist_id').size().cumsum().values ]
        tracks = test['actions'].track_id.values
        artists = test['actions'].artist_id.values
        
        done = 0
    
        for row in list( zip( test['playlists']['playlist_id'], test['playlists']['name'], test['playlists']['num_tracks'], test['playlists']['num_samples'] ) ):
            
            pid, name, ntracks, nsamples = row
            num_hidden = ntracks - nsamples
            
            if pid in plidmap:
                sidx = plidmap[pid]
                s = start[sidx]
                e = start[sidx+1]
                actions = tracks[s:e]
                artist_ids = artists[s:e]
            else:
                actions = None
                artist_ids = None
            
            tpredict = time.time()
            res = v.predict( name, actions, playlist_id=pid, artists=artist_ids, num_hidden=num_hidden )
            pt = time.time() - tpredict
            
            results[k]['playlist_id'] += [pid]*len(res)
            results[k]['track_id'] += list(res.track_id.values)
            results[k]['confidence'] += list(res.confidence.values)
            results_time[k] += pt
            
            done += 1
            
            if done % 100 == 0:
                print( ' -- finished {} of {} test lists in {}s'.format( done, len(test['playlists']), (time.time() - tstart) ) )

        results[k] = pd.DataFrame.from_dict( results[k] )
        fname = 'results'+str(NUM_RECOMMENDATIONS)+'_' if NUM_RECOMMENDATIONS != 500 else 'results_'
        results[k].to_csv( FOLDER_TEST + fname + k +'.csv' )
        print( 'prediction time for {}: {}'.format( k, (results_time[k] / len( test['playlists'] ) ) ) )
        
        algs[k] = None
        gc.collect()
        
    eval( algs.keys() )
     
def eval(list):
    
    preloaded = inout.load_validation(FOLDER_TEST)

    if preloaded is not None:
        preloaded[0].sort_values( ['num_samples','name'], inplace=True )
    
        all = pd.DataFrame()
        all_parts = pd.DataFrame()
    
        for m in list:
            res = pd.read_csv( FOLDER_TEST + 'results_'+m+'.csv' )
            res, res_parts = evaluate( res, FOLDER_TEST, strict=True, preloaded=preloaded )
            res['method'] = [m]
            res_parts['method'] = [m]
    
            all = pd.concat([ all, res ])
            all_parts = pd.concat([ all_parts, res_parts ])
    
        print( all )
        print( all_parts )
    
        all.to_csv( FOLDER_TEST + 'eval.csv' )
        all_parts.to_csv( FOLDER_TEST + 'eval_parts.csv' )
    
if __name__ == '__main__':
    main()
