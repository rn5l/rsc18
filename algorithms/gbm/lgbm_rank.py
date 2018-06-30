from _operator import itemgetter
from math import sqrt
import random
import time

import numpy as np
import pandas as pd

import lightgbm as lgb
from pip._vendor.colorama.initialise import init


class LGBMRank:
    '''
    LGBMRank( epochs=5, item_latent=None, session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key='artist_id', time_key= 'pos', folder='' )

    Parameters
    -----------
    '''
    
    def __init__( self, epochs=5, item_latent=None, session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key='artist_id', time_key= 'pos', folder='' ):
       
        self.epochs = epochs
        self.include_artist = False
        
        self.session_key = session_key
        self.item_key = item_key
        self.user_key = user_key
        self.time_key = time_key
        self.artist_key = artist_key
        self.cat_features = range(11)
        self.item_latent = item_latent
        
        self.folder = folder
        
        self.latent_features = None
        
        self.intX = np.int32
    
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
        
        
        start = time.time()
        
        self.unique_items = data[self.item_key].unique().astype( self.intX )
                
        self.num_items = data[self.item_key].nunique()
        self.num_users = data[self.user_key].nunique()
        self.num_artists = data[self.artist_key].nunique()
        #idx = [data[self.item_key].max()+1] + list( data[self.item_key].unique() )
        self.itemmap = pd.Series( data=np.arange(self.num_items), index=data[self.item_key].unique() ).astype( self.intX )
        self.usermap = pd.Series( data=np.arange(self.num_users), index=data[self.user_key].unique() ).astype( self.intX )
        #self.artistmap = pd.Series( data=np.arange(self.num_artists), index=data[self.artist_key].unique() ).astype( self.intX )

        print( 'finished init item and user map in {}'.format(  ( time.time() - start ) ) )
        
        
        train = data
        
        start = time.time()
                
        self.num_sessions = train[self.session_key].nunique()
        
        #train = pd.merge(train, pd.DataFrame({self.item_key:self.itemmap.index, 'ItemIdx':self.itemmap[self.itemmap.index].values}), on=self.item_key, how='inner')
        train = pd.merge(train, pd.DataFrame({self.user_key:self.usermap.index, 'UserIdx':self.usermap[self.usermap.index].values}), on=self.user_key, how='inner')
        #train = pd.merge(train, pd.DataFrame({self.artist_key:self.artistmap.index, 'ArtistIdx':self.artistmap[self.artistmap.index].values}), on=self.artist_key, how='inner')
        #train.sort_values([self.session_key, self.time_key], inplace=True)
        
        #self.predict_model = self.init_model( train )
        
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
           
        if self.epochs == 0:
            return
                
        start = time.time()
        
        max_list = train.UserIdx.max()
        split_list = max_list * 0.8
        
        traintr = train[ train.UserIdx < split_list ]
        trainvld = train[ train.UserIdx > split_list ]
        
        df, Qv = self.get_train_df( trainvld )
        if self.item_latent > 1:
            df = self.add_latent(df, size=self.item_latent, target=True, avg=True, last=True )
        yv = df['label'].values
        del df['label']
        Xv = df.values
        del df
        
#         lgbt = lgb.Dataset( X, label=y, group=Q, categorical_feature=self.cat_features )
#         lgbv = lgb.Dataset( Xv, label=Yv, group=Qv, categorical_feature=self.cat_features )
        
        params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'lambdarank',
        'metric': {'ndcg'},
        'num_leaves': 128,
        #'min_data_in_leaf':2,
        'learning_rate': 0.1,
        #'feature_fraction': 0.9,
        #'bagging_fraction': 0.8,
        #'bagging_freq': 5,
        'verbose': 1
        }
        
        #mdl = lgb.train( params, lgbt, valid_sets=lgbv, num_boost_round=50, early_stopping_rounds=20, verbose_eval=1 )
        
        df, Q = self.get_train_df( traintr, random=True )
        
        start = time.time()
        print(len(df))
        if self.item_latent > 1:
            df = self.add_latent(df, size=self.item_latent, target=True, avg=True, last=True )
        print(len(df))
        print( 'added latent factors in {}'.format(  ( time.time() - start ) ) )
        
        y = df['label'].values
        del df['label']
        X = df.values
        del df
        
        lgbt = lgb.Dataset( X, label=y, group=Q, categorical_feature=self.cat_features )
        lgbv = lgb.Dataset( Xv, label=yv, group=Qv, categorical_feature=self.cat_features, free_raw_data=False )
        
        mdl = lgb.train( params, lgbt, valid_sets=lgbv, num_boost_round=10, early_stopping_rounds=3, verbose_eval=1 )
        
        for j in range( 1,self.epochs ):
                 
            df, Q = self.get_train_df( traintr, random=True )
            if self.item_latent > 1:
                df = self.add_latent(df, size=self.item_latent, target=True, avg=True, last=True)
            y = df['label'].values
            del df['label']
            X = df.values
            del df
            
            #Xv, Yv, Qv = self.get_train_data( trainvld, random=True )
            lgbt = lgb.Dataset( X, label=y, group=Q, categorical_feature=self.cat_features )
            #lgbv = lgb.Dataset( Xv.T, label=Yv, group=Qv, categorical_feature=self.cat_features )
             
            #X = np.concatenate( [X,Xt] )
            #y = np.concatenate( [y,yt] )
            #Q = np.concatenate( [Q,Qt] )
             
            #print( 'train epoch {} with {} examples'.format( j, len(X) ) )
             
            mdl = lgb.train( params, lgbt, valid_sets=[lgbv], num_boost_round=10, early_stopping_rounds=3, verbose_eval=1, init_model=mdl )
             
            #print( 'finished epoch {} in {}s'.format( j, ( time.time() - start ) ) )
                
        #res = mdl.predict(Xv.T, num_iteration=mdl.best_iteration)
        
        self.predict_model = mdl
        
        self.pop_map = pd.DataFrame()
        self.pop_map['pop'] = train.groupby('track_id').size()
        self.pop_map['pop'] = self.pop_map['pop'] / self.pop_map['pop'].sum()
        

    def get_train_df( self, train, seed=5, random=False, neg_samples=1 ):
        
        train.sort_values( [ self.user_key,'pos' ], inplace=True )
        umap = pd.Series( index=train[self.user_key].unique(), data = range(train[self.user_key].nunique()) )
        start = np.r_[ 0, train.groupby(self.user_key).size().cumsum().values ]
        tracks = train[self.item_key].values
        artists = train[self.artist_key].values
        
        df_dict = {}
        for i in range(seed):
            df_dict['track_id_'+str(i)] = []
            df_dict['artist_id_'+str(i)] = []
        
        df_dict['target'] = []
        df_dict['label'] = []
        
        query = []
        
        for uidx in train[self.user_key].unique():
            
            uid = umap[uidx]
            s = start[uid]
            e = start[uid+1]
            actions = tracks[s:e]
            aactions = artists[s:e]
            
            if len( actions ) <= seed :
                continue
            
            mask = np.array( range(len( actions )) ) < seed
            if random:
                mask = np.random.permutation( mask )
            
            seedt = actions[mask]
            seeda = aactions[mask]
            positive = actions[~mask]
            negative = np.random.choice( self.unique_items, len(positive) )
            
            labelp = [1] * len(positive)
            labeln = [0] * len(negative)
            labelp = labelp + labeln
            
            positive = np.concatenate( (positive, negative) )
            
            seedt = np.tile( seedt, reps=(len( positive ), 1) ).T
            seeda = np.tile( seeda, reps=(len( positive ), 1) ).T
                        
            for i in range(seed):
                df_dict['track_id_'+str(i)].extend( seedt[i] )
                df_dict['artist_id_'+str(i)].extend( seeda[i] )
            df_dict['target'].extend( positive )
            df_dict['label'].extend( labelp )
            query += [ len( positive ) ]
        
        
        df = pd.DataFrame( df_dict )
        q = np.array( query )
        
        return df, q
    
    def add_latent( self, df, seed=5, type='als', size=32, col='track_id', count=False, target=False, avg=True, last=False ):
        
        if self.latent_features is None:
            self.latent_features = dict()
        
        key = type + '_' + col + '_features' + ( '_cnt' if count else '' ) + '.' + str(size) + '.csv'
        
        if not key in self.latent_features:
            self.latent_features[key] = pd.read_csv( self.folder + key )
            self.latent_features[key].index = self.latent_features[key][col]
        
        features = self.latent_features[key]
        LF = [ 'lf_'+str(i) for i in range(size) ]
        features = features[ LF + [col] ]
        
        if target:
            TLF = [ 'target_lf_'+str(i) for i in range(size) ]
            features.columns = TLF + [col]
            df = df.merge( features, how='inner', left_on='target', right_on=col )
            del df[col]
            
        if avg:
            for s in range(seed):
                TLF = [ col+'_avg_tmp_'+str(i) for i in range(size) ]
                features.columns = TLF + [col]
                df = df.merge( features, how='inner', left_on=col+'_'+str(s), right_on=col )
                del df[col]
                if not col+'_avg_'+str(0) in df.columns: 
                    for j in range(size):
                        df[ col+'_avg_'+str(j) ] = df[ col+'_avg_tmp_'+str(j) ]
                else:
                    for j in range(size):
                        df[ col+'_avg_'+str(j) ] += df[ col+'_avg_tmp_'+str(j) ]
                
                for k in TLF:
                    del df[ k ]
            for j in range(size):
                df[ col+'_avg_'+str(j) ] = df[ col+'_avg_'+str(j) ] / seed
            
        if last:
            LLF = [ col+'_last_lf_'+str(i) for i in range(size) ]
            features.columns = LLF + [col]
            df = df.merge( features, how='inner', left_on=col+'_'+str(seed-1), right_on=col )
            del df[col]
         
        return df
    
    def get_latent( self, ids, type='als', size=32, col='track_id', count=False, avg=True, last=False ):
        
        if self.latent_features is None:
            self.latent_features = dict()
        
        key = type + '_' + col + '_features' + ( '_cnt' if count else '' ) + '.' + str(size) + '.csv'
        
        if not key in self.latent_features:
            self.latent_features[key] = pd.read_csv( self.folder + key )
            self.latent_features[key].index = self.latent_features[key][col]
            
        features = self.latent_features[key]
        features.index = features[col]
                     
        if avg:
            ft = features.ix[ids]
            LF = [ 'lf_'+str(i) for i in range(size) ]
            res = ft[LF].mean().values
            
        if last:
            ft = features.ix[ids[-1]]
            LF = [ 'lf_'+str(i) for i in range(size) ]
            if res is not None:
                res = np.concatenate( [res, ft[LF].values] )
            else:
                res = ft[LF].values
         
        return res
    
    def get_all_latent( self, ids, type='als', size=32, col='track_id', count=False ):
        
        if self.latent_features is None:
            self.latent_features = dict()
        
        key = type + '_' + col + '_features' + ( '_cnt' if count else '' ) + '.' + str(size) + '.csv'
        
        if not key in self.latent_features:
            self.latent_features[key] = pd.read_csv( self.folder + key )
            self.latent_features[key].index = self.latent_features[key][col]
        
        features = self.latent_features[key]
        #features.index = features[col]
                        
        ft = features.ix[ids]
        LF = [ 'lf_'+str(i) for i in range(size) ]
        
        return ft[LF].values
                                                                        
    def predict( self, name=None, tracks=None, playlist_id=None, artists=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        name : int or string
            The session IDs of the event.
        tracks : int list
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        res : pandas.DataFrame
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        sitems = tracks if tracks is not None else []
        
        if len(sitems) < 5:
            res_dict = {}
            res_dict['track_id'] =  []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)
           
        #tss = time.time()     
        last_five = sitems[-5:]
        #last_five = self.itemmap[ last_five ]
        seed_tracks = np.array( last_five )
        seed_tracks = np.tile( seed_tracks, reps=(self.num_items,1) )
        #print( 'seed stack in {}s'.format( ( time.time() - tss ) ) )
        
        if self.item_latent > 1:
            seed_latent = self.get_latent(last_five, size=self.item_latent, col='track_id', avg=True, last=True)
            seed_latent = np.tile( seed_latent, reps=(self.num_items,1) )
        
        last_five = artists[-5:]
        seed_artists = np.array( last_five )
        seed_artists = np.tile( seed_artists, reps=(self.num_items,1) )      
        
        #tss = time.time()     
        i = np.array( [self.itemmap.index] )
        
        if self.item_latent > 1:
            target_latent = self.get_all_latent( self.itemmap.index, size=self.item_latent )
        
            data = np.concatenate( (seed_artists.T,i,seed_tracks.T,target_latent.T, seed_latent.T  ) )
        else:
            data = np.concatenate( (seed_artists.T,i,seed_tracks.T) )

        #print( 'concat in {}s'.format( ( time.time() - tss ) ) )
#         usera = np.zeros((1))
#         usera[0] = self.usermap[input_user_id]
        
        #tss = time.time()     
        predictions = self.predict_model.predict( data.T, num_iteration=self.predict_model.best_iteration ) #, usera ] )
        #predictions = self.predict( self.session_items, self.itemmap[input_item_id], self.usermap[input_user_id] )
        
        #print( 'prediction in {}s'.format( ( time.time() - tss ) ) )
                
        try:
            #tss = time.time()   
            
            # Create things in the format
            res_dict = {}
            res_dict['track_id'] =  list(self.itemmap.index)
            res_dict['confidence'] = predictions
            res = pd.DataFrame.from_dict(res_dict)
            res = res[ ~np.in1d( res.track_id, sitems ) ]
            
            res['confidence'] += self.pop_map['pop'][res.track_id].values * 0.1
            
            res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )  
                                    
            #print( 'format + sort in {}s'.format( ( time.time() - tss ) ) ) 
            
        except Exception:
            print( 'hö' )
            print( self.itemmap.index )
            print( predictions )
            print( len(predictions[0]) )
            exit()
                
        return res.head(500)
        
