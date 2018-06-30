'''
Created on 13.04.2018

@author: malte
'''
import gc
import pickle
import random
import time

from matplotlib.dviread import Page

from algorithms.baseline.Popular import MostPopular
from algorithms.hybrid.Fill import Fill
from algorithms.mf.implicitu import Implicit
from helper import inout
from helper.eval import evaluate
import numpy as np
import pandas as pd


NUM_RECOMMENDATIONS=500

# data folder
FOLDER_TRAIN = 'data/data_formatted_50k/'
FOLDER_TEST = 'data/data_formatted_50k/'

def main():
    
    train, test = inout.load_dataset( FOLDER_TRAIN, FOLDER_TEST, feather=True )
    
    export_csv_base = 'opt_implicit/test'

    factors = [ 32, 50, 64,100,150,200,300,500 ]
    reg = list(np.linspace(0.001, 0.01, num=10)) + list(np.linspace(0.01, 0.1, num=10 ))
    epochs = [ 5,10,15,20,25,50 ]
    filter = [ 5,10,20,30 ]
    
    best = 0
    bestp = 0
    bestn = 0
    best_key = ""
    
    for i in range(100):
        
        gc.collect()
        
        print( 'opt in iteration {}'.format(i) )
        
        algs = {}
        
        nfact = random.choice(factors)
        vreg = random.choice(reg)
        nits = random.choice(epochs)
        ifilter = random.choice(filter)
        
        key = 'implicit-{}f-{}e-{}reg-filter20.{}'.format(nfact,nits,vreg,ifilter)
        
        algs[key] = Fill( Implicit( nfact, nits, reg=vreg, filter=(20,ifilter) ), MostPopular() )
        
        print( ' -- current run for {}:'.format(key) )
        
        for k, v in algs.items():
            tstart = time.time()
            v.train( train, test=test )
            print( ' -- trained {} in {}s'.format( k, (time.time() - tstart) ) )
        
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
        
        key = ""
        for k, v in algs.items():
            results[k] = pd.DataFrame.from_dict( results[k] )
            results[k].to_csv( FOLDER_TEST + export_csv_base + k + '_'+str(i)+'.csv' )
            print( 'prediction time for {}: {}'.format( k, (results_time[k] / len( test['playlists'] ) ) ) )
        
            key=k
        rp, page, ndcg = eval( algs.keys(), export_csv_base, i )
        
        if rp > best:  # new best found
                best = rp
                bestp = page
                bestn = ndcg
                best_key = key
                
        print('CURRENT BEST: ' + best_key)
        print('WITH RP@500: ' + str(best))
        print('WITH PAGE@500: ' + str(bestp))
        print('WITH NDCG@500: ' + str(bestn))
        
     
def eval(list, basepath, iteration):
    
    preloaded = inout.load_validation(FOLDER_TEST)
    preloaded[0].sort_values( ['num_samples','name'], inplace=True )
    
    all = pd.DataFrame()
    all_parts = pd.DataFrame()
    
    for m in list:
        res = pd.read_csv( FOLDER_TEST + basepath + m + '_'+str(iteration)+'.csv' )
        res, res_parts = evaluate( res, FOLDER_TEST, strict=True, preloaded=preloaded )
        res['method'] = [m]
        res_parts['method'] = [m]
    
        all = pd.concat([ all, res ])
        all_parts = pd.concat([ all_parts, res_parts ])
    
    print( all )
    #print( all_parts )
    
    all.to_csv( FOLDER_TEST + basepath + '_eval_'+str(iteration)+'.csv' )
    all_parts.to_csv( FOLDER_TEST + basepath + '_evalparts_'+str(iteration)+'.csv' )
    
    return all['rp'].values[0], all['pages'].values[0], all['ndcg'].values[0]

if __name__ == '__main__':
    main()
    