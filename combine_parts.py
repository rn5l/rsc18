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
from algorithms.rerank.MetaRerank import MetaRerank
from algorithms.string_matching.implicit_match import ImplicitStringMatch
from algorithms.string_matching.string_matching import StringMatching
from helper import inout
from helper.eval import evaluate
import numpy as np
import pandas as pd


NUM_RECOMMENDATIONS=500

# data folder
FOLDER_TRAIN = 'data/data_formatted_50k/'
FOLDER_TEST = 'data/sample_50k_similar/'

def main():
    
    algs = {}
    
    #combine algorithms in a hybrid
    
    mp = MostPopular()
    
    diskknn = Solution( FOLDER_TEST + 'results_diskknn.csv' )
    sknn = Solution( FOLDER_TEST + 'results_sknn.csv' )
    iknn = Solution( FOLDER_TEST + 'results_iknn.csv' )
    implicit = Solution( FOLDER_TEST + 'results_implicit.csv' )
    
    smatch = Solution( FOLDER_TEST + 'results_smatch.csv' )
    imatch = Solution( FOLDER_TEST + 'results_imatch.csv' )
    
    titlerec = Weighted( [smatch,imatch], [0.5,0.5] ) #best one for title only
    hybrid = Weighted( [diskknn,iknn,sknn,implicit], [0.4,0.3,0.2,0.1] ) #best one for lists with 5+ seed tracks
    firstcat = Weighted( [hybrid,titlerec], [0.7,0.3] ) #best one for lists with only one seed track
    
    algs['recommender'] = Switch( [titlerec,firstcat,hybrid], [1,5,101] ) #main
    #algs['meta-recommender'] = MetaRerank( Switch( [titlerec,firstcat,hybrid], [1,5,101] ) ) #creative
    
    #to make the process more flexible, all methods have been trained and reused the following way:
#     #first run
#     algs['myname-with-params'] = Fill( SessionKNN( 2000, 0, idf_weight=1 ), mp )
#     #reuse predictions
#     algs['myname-with-params'] = Solution( FOLDER_TEST + 'results_myname-with-params.csv' )
    
    #start processing
    
    train, test = inout.load_dataset( FOLDER_TRAIN, FOLDER_TEST, feather=True )
    
    for k, v in algs.items():
        tstart = time.time()
        v.train( train, test=test )
        print( 'trained {} in {}s'.format( k, (time.time() - tstart) ) )
    
    results = {}
    results_time = {}

    for k, v in algs.items():
        results[k] = {}
        results[k]['playlist_id'] = []
        results[k]['track_id'] = []
        results[k]['confidence'] = []
        results_time[k] = 0
    
    tstart = time.time()
        
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
        
        for k, v in algs.items():
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

    for k, v in algs.items():
        results[k] = pd.DataFrame.from_dict( results[k] )
        fname = 'results'+str(NUM_RECOMMENDATIONS)+'_' if NUM_RECOMMENDATIONS != 500 else 'results_'
        results[k].to_csv( FOLDER_TEST + fname + k +'.csv' )
        print( 'prediction time for {}: {}'.format( k, (results_time[k] / len( test['playlists'] ) ) ) )

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
    #eval( ['sknn-500-5000-artist2' ] ) #, 'ar-100', 'sr-100', 'iknn-100'] )
