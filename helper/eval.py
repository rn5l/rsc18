'''
Created on 11.04.2018

@author: malte
'''
from helper import inout
import numpy as np
import pandas as pd
import math
import time
from collections import OrderedDict

FOLDER_TEST = '../data/sample_similar/'
FOLDER_TEST = '../data/sample_100k_random/'
FOLDER_TEST = '/media/malte/Datastorage/mpd/mpd-share/sample3_random/'

def evaluate( result, test_folder, strict=False, preloaded=None ):
    
    print( 'evaluate result ',( 'strict' if strict else 'loose' ) )
    
    if preloaded is None:
        lists, truth = inout.load_validation(test_folder)
        lists.sort_values( 'num_samples', inplace=True )
    else:
        lists, truth = preloaded
    
    print( 'number of test lists: ', len(lists) )
    not_in_actions = np.setdiff1d( lists.playlist_id.unique(), truth.playlist_id.unique() )
    if len( not_in_actions ) > 0 :
        print( lists[ np.in1d(lists.playlist_id, not_in_actions) ] )
        print( 'validation data missing' )
        exit()
    
    print( ' -- eval set loaded' )
    
    res = pd.DataFrame()
    res['rp'] = [0]
    res['pages'] = [0]
    res['ndcg'] = [0]
    
    res_parts = pd.DataFrame()
    
    count = 0
    tstart = time.time()
    
    result.reset_index(inplace=True)
    result_map = pd.Series( index=list(result.playlist_id.unique()), data=range(len(result.playlist_id.unique())) )
    result_start = np.r_[ 0, result.groupby('playlist_id').size().cumsum().values ]
    result_tracks = result.track_id.values
    
    truth.sort_values( ['playlist_id','pos'], inplace=True )
    truth = truth.reset_index(drop=True)
    true_map = pd.Series( index=list(truth.playlist_id.unique()), data=range(len(truth.playlist_id.unique())) )
    true_start = np.r_[ 0, truth.groupby('playlist_id').size().cumsum().values ]
    true_tracks = truth.track_id.values
    
    klist = []
    
    for plist in lists.itertuples():
        
        pid = plist.playlist_id
        
        if pid not in result_map.index:
            if strict:
                print( 'no results for playlist ', pid )
                exit()
            continue
        
        true_idx = true_map[pid]
        result_idx = result_map[pid]
        
        
            
        
        recs = result_tracks[ result_start[result_idx] : result_start[result_idx + 1] ]
        tracks = true_tracks[ true_start[true_idx] : true_start[true_idx + 1] ]
                
        if len(tracks) == 0:
            print( 'no tracks for playlist ', plist.playlist_id )
            exit()
        
        if strict and len(recs) != 500:
            print( recs )
            print( len( recs ) )
            raise Exception( 'no valid result set for playlist ', plist.playlist_id )
        elif len( recs ) == 0:
            continue
        
        k = key(plist)
        if not 'rp_'+k in res_parts.columns:
            klist.append(k)
            res_parts['rp_'+k] = [0]
            res_parts['page_'+k] = [0]
            res_parts['ndcg_'+k] = [0]
            res_parts['count_'+k] = [0]
            res_parts['samples_'+k] = [plist.num_samples]
            
        r_prec = r_precision( recs, tracks )
        pages = rec_page( recs, tracks )
        ndcga = ndcg( recs, tracks )
        
        res['rp'] += r_prec
        res['pages'] += pages
        res['ndcg'] += ndcga
        
        res_parts['rp_'+k] += r_prec
        res_parts['page_'+k] += pages
        res_parts['ndcg_'+k] += ndcga
        
        count += 1
        res_parts['count_'+k] += 1
        
        if count % 5000 is 0: 
            print( ' -- evaluated {} of {} lists in {}s'.format( count, len(lists), (time.time() - tstart) ) )
       
    print( ' -- evaluated all lists in {}s'.format( (time.time() - tstart) ) ) 
    
    res = res / count
    for k in klist:
        res_parts['rp_'+k] = res_parts['rp_'+k] / res_parts['count_'+k] 
        res_parts['page_'+k] = res_parts['page_'+k] / res_parts['count_'+k] 
        res_parts['ndcg_'+k]  = res_parts['ndcg_'+k] / res_parts['count_'+k]
    
    return res, res_parts
        
def r_precision( rec, actual ):    
    actual = np.unique( actual )
    mask = np.in1d( rec[:len(actual)], actual )
    res = ( mask.sum() / len(actual) )
    if math.isnan(res):
        print( 'r_prec is nan' )
        print( mask.sum() )
        print( actual )
        print( rec )
        print( mask )
        print( res )
    if res > 1:
        print( 'r_prec is bigger than one' )
        print( mask.sum() )
        print( actual )
        print( rec )
        print( mask )
        print( res )
    return res

def rec_page( rec, tracks ):   
    res = 51
    mask = np.in1d( rec, tracks )
    hits = np.arange(len(rec))[mask]
    if len(hits) > 0:
        first_hit = hits.min()
        res = math.floor( first_hit / 10 )
    return res

def dcg( rec, actual, k=500 ):
    
    rel = np.in1d( rec[:k], actual ) * 1.
    #log = np.log2( np.arange( 1, len(rec)+1 ) )
    #dcg = (rel[0] / 1) + np.sum( rel[1:] / log[1:] )
    dcg = np.sum(rel / np.log2(1 + np.arange(1, k + 1)))
    
    return dcg

def ndcg( rec, actual ):    
    
    actual = np.unique( actual )
    
    cut = np.in1d( actual, rec ).sum()
    if cut == 0:
        return 0.
    
#     rel = np.ones( len(actual) )
#     log = np.log2( np.arange(1,len(actual)+1) )
#     idcg = rel[0] / 1 + np.sum( rel[1:] / log[1:] )
    idcg = dcg( actual, actual, min( len(rec), len(actual) ) )
    rdcg = dcg( rec, actual )
    
    return rdcg / idcg

def key(plist):
    return str(plist.num_samples) + ('' if plist.name is None or plist.name == '' and plist.name == 'nan' or type(plist.name) is float else 't') + ('o' if plist.in_order else '')

def r_precision_pl(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    target_set = set(targets)
    target_count = len(target_set)
    return float(len(set(predictions[:target_count]).intersection(target_set))) / target_count

def rec_page_pl(targets, predictions, max_n_predictions=500):
    # Assumes predictions are sorted by relevance
    # First, cap the number of predictions
    predictions = predictions[:max_n_predictions]

    # Calculate metric
    i = set(predictions).intersection(set(targets))
    for index, t in enumerate(predictions):
        for track in i:
            if t == track:
                return float(int(index / 10))
    return float(max_n_predictions / 10.0 + 1)

def ndcg_pl(relevant_elements, retrieved_elements, k=500, *args, **kwargs):
    r"""Compute the Normalized Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    The metric is determined by calculating the DCG and dividing it by the
    ideal or optimal DCG in the case that all recommended tracks are relevant.
    Note:
    The ideal DCG or IDCG is on our case equal to:
    \[ IDCG = 1+\sum_{i=2}^{min(\left| G \right|, k)}\frac{1}{\log_2(i +1)}\]
    If the size of the set intersection of \( G \) and \( R \), is empty, then
    the IDCG is equal to 0. The NDCG metric is now calculated as:
    \[ NDCG = \frac{DCG}{IDCG + \delta} \]
    with \( \delta \) a (very) small constant.
    The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Returns:
        NDCG value
    """

    # TODO: When https://github.com/scikit-learn/scikit-learn/pull/9951 is
    # merged...
    idcg = dcg_pl(
        relevant_elements, relevant_elements, min(k, len(relevant_elements)))
    if idcg == 0:
        raise ValueError("relevent_elements is empty, the metric is"
                         "not defined")
    true_dcg = dcg_pl(relevant_elements, retrieved_elements, k)
    return true_dcg / idcg

def dcg_pl(relevant_elements, retrieved_elements, k, *args, **kwargs):
    """Compute the Discounted Cumulative Gain.
    Rewards elements being retrieved in descending order of relevance.
    \[ DCG = rel_1 + \sum_{i=2}^{|R|} \frac{rel_i}{\log_2(i + 1)} \]
    Args:
        retrieved_elements (list): List of retrieved elements
        relevant_elements (list): List of relevant elements
        k (int): 1-based index of the maximum element in retrieved_elements
        taken in the computation
    Note: The vector `retrieved_elements` is truncated at first, THEN
    deduplication is done, keeping only the first occurence of each element.
    Returns:
        DCG value
    """
    retrieved_elements = __get_unique(retrieved_elements[:k])
    relevant_elements = __get_unique(relevant_elements)
    if len(retrieved_elements) == 0 or len(relevant_elements) == 0:
        return 0.0
    # Computes an ordered vector of 1.0 and 0.0
    score = [float(el in relevant_elements) for el in retrieved_elements]
    # return score[0] + np.sum(score[1:] / np.log2(
    #     1 + np.arange(2, len(score) + 1)))
    return np.sum(score / np.log2(1 + np.arange(1, len(score) + 1)))

def __get_unique(original_list):
    """Get only unique values of a list but keep the order of the first
    occurence of each element
    """ 
    return list(OrderedDict.fromkeys(original_list))

if __name__ == '__main__':
    
    results_vknn = pd.read_csv( FOLDER_TEST + 'results_sknn-500-5000.csv' )

    res, res_parts = evaluate( results_vknn, FOLDER_TEST, strict=True )
    print(res)

    #inout.save_submission( results_mp, FOLDER_TEST + 'submission_mp.csv' )
    
