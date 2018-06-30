# -*- coding: utf-8 -*-
"""
Created on 09.06.2018
Based on https://github.com/dawenl/vae_cf/blob/master/VAE_ML20M_WWW2018.ipynb
@author: malte
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import sparse
import bottleneck as bn
from algorithms.ae.helper.dae import MultiDAE
from algorithms.ae.helper.vae import MultiVAE
import sys
import time

class AutoEncoder:
    '''
    AutoEncoder(layers=[200,600], epochs = 10, lr = 0.01, algo='vae', session_key = 'playlist_id', item_key = 'track_id', folder='')
    
    CF with variational and denoising auto encoders
        
    Parameters
    --------
    layers : []
        Layers sizes for encoding, which will be mirrored for decoding.
    epochs : int
        The number of epoch for training. (Default value: 10)
    learning_rate : float
        Learning rate. (Default value: 0.01)
    algo : string
        "vae" or "dae" depending on the approach which should be applied.
    lambda_item : float
        Regularization for item features. (Default value: 0.0)
    session_key : string
        Header of the session ID column in the input file (default: 'playlist_id')
    item_key : string
        Header of the item ID column in the input file (default: 'track_id')
    folder : string
        Folder to save the model in.
    
    '''
    def __init__(self, layers=[200,600], epochs = 10, lr = 0.01, algo='vae', session_key = 'playlist_id', item_key = 'track_id', folder=''):
        self.layers = layers
        self.epochs = epochs
        self.lr = lr
        self.folder = folder + 'aemodel'
        
        self.algo = algo
        self.session_key = session_key
        self.item_key = item_key
    
    def train(self, train, test=None):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        data = train['actions']
            
        data = self.filter_data(data,min_uc=5,min_sc=5)    
            
        itemids = data[self.item_key].unique()
        self.n_items = len(itemids)
        self.itemidmap = pd.Series(data=np.arange(self.n_items), index=itemids)
        self.itemidmap2 = pd.Series(index=np.arange(self.n_items), data=itemids)
        self.predvec = np.zeros( self.n_items )
        
        sessionids = data[self.session_key].unique()
        self.n_sessions = len(sessionids)
        self.useridmap = pd.Series(data=np.arange(self.n_sessions), index=sessionids)
        
        data = pd.merge(data, pd.DataFrame({self.item_key:self.itemidmap.index, 'iid':self.itemidmap[self.itemidmap.index].values}), on=self.item_key, how='inner')
        data = pd.merge(data, pd.DataFrame({self.session_key:self.useridmap.index, 'uid':self.useridmap[self.useridmap.index].values}), on=self.session_key, how='inner')
                        
        n_kept_users = int( self.n_sessions * 0.9 )
        
        data_val = data[ data.uid > n_kept_users ]
        data = data[ data.uid <= n_kept_users ]
        
        ones = np.ones( len(data) )
        col_ind = self.itemidmap[ data.track_id.values ]
        row_ind = self.useridmap[ data.playlist_id.values ] 
        mat = sparse.csr_matrix((ones, (row_ind, col_ind)), shape=(self.n_sessions, self.n_items))
        
        data_val_tr, data_val_te = self.split_train_test_proportion( data_val )
                
        data_val_tr['uid'] = self.useridmap[ data_val_tr.playlist_id.values ].values
        data_val_tr['uid'] = data_val_tr['uid'] - data_val_tr['uid'].min() 
        ones = np.ones( len(data_val_tr) )
        col_ind = self.itemidmap[ data_val_tr.track_id.values ]
        row_ind = data_val_tr.uid.values
        mat_val_tr = sparse.csr_matrix((ones, (row_ind, col_ind)) , shape=( data_val_tr.playlist_id.nunique() , self.n_items))
        
        data_val_te['uid'] = self.useridmap[ data_val_te.playlist_id.values ].values
        data_val_te['uid'] = data_val_te['uid'] - data_val_te['uid'].min() 
        ones = np.ones( len(data_val_te) )
        col_ind = self.itemidmap[ data_val_te.track_id.values ]
        row_ind = data_val_te.uid.values
        mat_val_te = sparse.csr_matrix((ones, (row_ind, col_ind)), shape=( data_val_te.playlist_id.nunique() , self.n_items))
        
        self.layers = self.layers + [self.n_items]
        
        if self.algo == 'dae':
            ae = MultiDAE( self.layers, q_dims=None, lam=0.01, lr=self.lr )
        elif self.algo == 'vae': 
            ae = MultiVAE( self.layers, q_dims=None, lam=0.01, lr=self.lr )

        self.model = ae
        
        N = mat.shape[0]
        idxlist = np.array( range(N) )
        
        # training batch size
        batch_size = 50
        batches_per_epoch = int(np.ceil(float(N) / batch_size))
        
        N_vad = mat_val_tr.shape[0]
        idxlist_vad = np.array( range(N_vad) )
        
        # validation batch size (since the entire validation set might not fit into GPU memory)
        batch_size_vad = 50
        
        # the total number of gradient updates for annealing
        total_anneal_steps = 200000
        # largest annealing parameter
        anneal_cap = 0.2
        
        saver, logits_var, loss_var, train_op_var, merged_var = ae.build_graph()
        
        ndcg_var = tf.Variable(0.0)
        ndcg_dist_var = tf.placeholder(dtype=tf.float64, shape=None)
        ndcg_summary = tf.summary.scalar('ndcg_at_k_validation', ndcg_var)
        ndcg_dist_summary = tf.summary.histogram('ndcg_at_k_hist_validation', ndcg_dist_var)
        merged_valid = tf.summary.merge([ndcg_summary, ndcg_dist_summary])
        
        summary_writer = tf.summary.FileWriter(self.folder, graph=tf.get_default_graph())

        ndcgs_vad = []

        with tf.Session() as sess:
        
            init = tf.global_variables_initializer()
            sess.run(init)
        
            best_ndcg = -np.inf
        
            update_count = 0.0
            
            tstart = time.time()
            
            for epoch in range(self.epochs):
                np.random.shuffle(idxlist)
                # train for one epoch
                for bnum, st_idx in enumerate(range(0, N, batch_size)):
                    end_idx = min(st_idx + batch_size, N)
                    X = mat[idxlist[st_idx:end_idx]]
                    
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')           
                    
                    if total_anneal_steps > 0:
                        anneal = min(anneal_cap, 1. * update_count / total_anneal_steps)
                    else:
                        anneal = anneal_cap
                    
                    feed_dict = {ae.input_ph: X, 
                                 ae.keep_prob_ph: 0.5, 
                                 ae.anneal_ph: anneal,
                                 ae.is_training_ph: 1}        
                    sess.run(train_op_var, feed_dict=feed_dict)
        
                    if bnum % 100 == 0:
                        summary_train = sess.run(merged_var, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_train, 
                                                   global_step=epoch * batches_per_epoch + bnum) 
                        
                        print( 'finished {} of {} in epoch {} in {}s'.format( bnum, batches_per_epoch, epoch, ( time.time() - tstart ) ) )
                    
                    update_count += 1
                
                # compute validation NDCG
                ndcg_dist = []
                for bnum, st_idx in enumerate(range(0, N_vad, batch_size_vad)):
                    end_idx = min(st_idx + batch_size_vad, N_vad)
                    X = mat_val_tr[idxlist_vad[st_idx:end_idx]]
        
                    if sparse.isspmatrix(X):
                        X = X.toarray()
                    X = X.astype('float32')
                    
                    print( X.sum(axis=1) )
                    print( X.sum(axis=0) )
                    
                    pred_val = sess.run(logits_var, feed_dict={ae.input_ph: X} )
                    # exclude examples from training and validation (if any)
                    pred_val[X.nonzero()] = -np.inf
                    ndcg_dist.append(self.NDCG_binary_at_k_batch(pred_val, mat_val_te[idxlist_vad[st_idx:end_idx]]))
                
                print(ndcg_dist)
                
                ndcg_dist = np.concatenate(ndcg_dist)
                ndcg_ = ndcg_dist.mean()
                ndcgs_vad.append(ndcg_)
                merged_valid_val = sess.run(merged_valid, feed_dict={ndcg_var: ndcg_, ndcg_dist_var: ndcg_dist})
                summary_writer.add_summary(merged_valid_val, epoch)
        
                # update the best model (if necessary)
                if ndcg_ > best_ndcg:
                    saver.save(sess, '{}/model'.format(self.folder))
                    best_ndcg = ndcg_
                
                print( 'finished epoch {} in {}s'.format( epoch, ( time.time() - tstart ) ) )
        
    def filter_data(self, data, min_uc=5, min_sc=0):
        # Only keep the triplets for items which were clicked on by at least min_sc users. 
        if min_sc > 0:
            itemcount = data[[self.item_key]].groupby(self.item_key).size()
            data = data[data[self.item_key].isin(itemcount.index[itemcount.values >= min_sc])]
        
        # Only keep the triplets for users who clicked on at least min_uc items
        # After doing this, some of the items will have less than min_uc users, but should only be a small proportion
        if min_uc > 0:
            usercount = data[[self.session_key]].groupby(self.session_key).size()
            data = data[data[self.session_key].isin(usercount.index[usercount.values >= min_uc])]
        
        return data
    
    def split_train_test_proportion(self, data, test_prop=0.2):
        
        data_grouped_by_user = data.groupby( self.session_key )
        tr_list, te_list = list(), list()
    
        np.random.seed(98765)
    
        for i, (_, group) in enumerate(data_grouped_by_user):
            n_items_u = len(group)
    
            if n_items_u >= 5:
                idx = np.zeros(n_items_u, dtype='bool')
                idx[np.random.choice(n_items_u, size=int(test_prop * n_items_u), replace=False).astype('int64')] = True
    
                tr_list.append(group[np.logical_not(idx)])
                te_list.append(group[idx])
            else:
                tr_list.append(group)
    
            if i % 1000 == 0:
                print("%d users sampled" % i)
                sys.stdout.flush()
        
        data_tr = pd.concat(tr_list)
        data_te = pd.concat(te_list)
        
        return data_tr, data_te
    
    def predict( self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : string
            playlist name
        tracks : int list
            tracks in the current playlist
        playlist_id : int
            playlist identifier
        artists : int list
            artists in the current playlist (per track)
        num_hidden : int
            Number of hidden tracks in the list
            
        Returns
        --------
        res : pandas.DataFrame
            sorted predictions with track_id and confidence
        
        '''
        
        items = tracks if tracks is not None else []
                    
            
        if len(items) == 0:
            res_dict = {}
            res_dict['track_id'] = []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)
            
        self.predvec.fill(0)
        self.predvec[ items ] = 1
        
        tf.reset_default_graph()
        saver, logits_var, _, _, _ = self.model.build_graph()
        
        with tf.Session() as sess:
            saver.restore(sess, '{}/model'.format(self.folder))
            recommendations = sess.run(logits_var, feed_dict={self.model.input_ph: self.predvec})
            
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] =  self.itemidmap2[ [x[0] for x in recommendations] ]
        res_dict['confidence'] = [x[1] for x in recommendations]
        res = pd.DataFrame.from_dict(res_dict)
        res.sort_values( 'confidence', ascending=False, inplace=True )
        
        return res.head(500)
    
    
    def NDCG_binary_at_k_batch(self, X_pred, heldout_batch, k=100):
        '''
        normalized discounted cumulative gain@k for binary relevance
        ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
        '''
        batch_users = X_pred.shape[0]
        idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
        topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                           idx_topk_part[:, :k]]
        idx_part = np.argsort(-topk_part, axis=1)
        # X_pred[np.arange(batch_users)[:, np.newaxis], idx_topk] is the sorted
        # topk predicted score
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]
        # build the discount template
        tp = 1. / np.log2(np.arange(2, k + 2))
    
        DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                             idx_topk].toarray() * tp).sum(axis=1)
        IDCG = np.array([(tp[:min(n, k)]).sum()
                         for n in heldout_batch.getnnz(axis=1)])
        return DCG / IDCG
        