from _operator import itemgetter
from datetime import datetime as dt
from datetime import timedelta as td
from math import log10
from math import sqrt
import random
import time

import keras
from keras.layers.embeddings import Embedding
from keras.regularizers import l2
from pympler import asizeof
import scipy.sparse
from scipy.sparse.csc import csc_matrix
import theano

import keras.backend as K
import keras.layers as kl
import keras.models as km
import numpy as np
import pandas as pd
import theano.tensor as T


class BPRNeuralCollaborativeFiltering:
    '''
    BPRNeuralCollaborativeFiltering( factors=8, layers=[64,32,16,8], batch=100, optimizer='adam', learning_rate=0.001, momentum=0.0, reg=0.01, emb_reg=1e-7, layer_reg=1e-7, dropout=0.0, skip=0, samples=2048, activation='linear', objective='bpr_max', epochs=10, shuffle=-1, include_artist=False, session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key='artist_id', time_key= 'pos'  )

    Parameters
    -----------
    '''
    
    def __init__( self, factors=8, layers=[64,32,16,8], batch=100, optimizer='adam', learning_rate=0.001, momentum=0.0, reg=0.01, emb_reg=1e-7, layer_reg=1e-7, dropout=0.0, skip=0, samples=2048, activation='linear', objective='bpr_max', epochs=10, shuffle=-1, include_artist=False, session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key='artist_id', time_key= 'pos' ):
       
        self.factors = factors
        self.layers = layers
        self.batch = batch
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = optimizer
        self.regularization = reg
        self.samples = samples
        self.dropout = dropout
        self.skip = skip
        self.shuffle = shuffle
        self.epochs = epochs
        self.activation = activation
        self.objective = objective
        self.include_artist = include_artist
        
        self.emb_reg = emb_reg
        self.layer_reg = layer_reg
        self.final_reg = reg
        
        self.session_key = session_key
        self.item_key = item_key
        self.user_key = user_key
        self.artist_key = artist_key
        self.time_key = time_key
                
        self.floatX = theano.config.floatX
        self.intX = 'int32'
        
    
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
        datat = test['actions']
        
        data = pd.concat( [data, datat] )
        
        start = time.time()
        
        self.unique_items = data[self.item_key].unique().astype( self.intX )
                
        self.num_items = data[self.item_key].nunique()
        self.num_users = data[self.user_key].nunique()
        self.num_artists = data[self.artist_key].nunique()
        #idx = [data[self.item_key].max()+1] + list( data[self.item_key].unique() )
        self.itemmap = pd.Series( data=np.arange(self.num_items), index=data[self.item_key].unique() ).astype( self.intX )
        self.usermap = pd.Series( data=np.arange(self.num_users), index=data[self.user_key].unique() ).astype( self.intX )
        self.artistmap = pd.Series( data=np.arange(self.num_artists), index=data[self.artist_key].unique() ).astype( self.intX )

        print( 'finished init item and user map in {}'.format(  ( time.time() - start ) ) )
        
        train = data
        
        start = time.time()
                
        self.num_sessions = train[self.session_key].nunique()
        
        train = pd.merge(train, pd.DataFrame({self.item_key:self.itemmap.index, 'ItemIdx':self.itemmap[self.itemmap.index].values}), on=self.item_key, how='inner')
        train = pd.merge(train, pd.DataFrame({self.user_key:self.usermap.index, 'UserIdx':self.usermap[self.usermap.index].values}), on=self.user_key, how='inner')
        train = pd.merge(train, pd.DataFrame({self.artist_key:self.artistmap.index, 'ArtistIdx':self.artistmap[self.artistmap.index].values}), on=self.artist_key, how='inner')
        #train.sort_values([self.session_key, self.time_key], inplace=True)
        
        self.itemartistmap = train.groupby( 'ItemIdx' )['ArtistIdx'].min()
        self.itemartistmap = pd.Series( index=self.itemartistmap.index, data = self.itemartistmap.values )
        
        self.model, self.predict_model = self.init_model( train )
        
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
                
        start = time.time()
        
        for j in range( self.epochs ):
            
            starttmp = time.time()
            
            U, I, N, A, AN = self.get_train_data( train )
            
            print( 'finished creating samples in {}'.format(  ( time.time() - starttmp ) ) )
            
            print( 'train epoch {} with {} examples'.format( j, len(U) ) )
            
            input = [np.array(U), np.array(I), np.array(N)]
            if self.include_artist:
                input += [ np.array(A), np.array(AN) ]
            
            hist = self.model.fit(input, #input
                         None, # labels 
                         batch_size=self.batch, epochs=1, shuffle=True, verbose=2 )
            
            print( 'finished epoch {} in {}s'.format( j, ( time.time() - start ) ) )
     
    def get_train_data( self, train ):
             
        #train = train.sample(frac=1).reset_index(drop=True)
        
        train['ItemIdxNeg'] = np.random.choice( self.itemmap.values, len(train) )
        items = train['ItemIdxNeg'].values
        train['ArtistIdxNeg'] = self.itemartistmap[ items ].values
        return train['UserIdx'].values, train['ItemIdx'].values, train['ItemIdxNeg'].values, train['ArtistIdx'].values, train['ArtistIdxNeg'].values
       
    def init_model(self, train, std=0.01):
        
        #current_item = kl.Input( ( 1, ), name="current_item" )
        
        item = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.,self.steps) )
        user = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        if self.include_artist:
            artist = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        emb_user_mf = Embedding( output_dim=self.factors, input_dim=self.num_users, embeddings_regularizer=l2(self.emb_reg) )
        emb_user = Embedding( output_dim=self.factors, input_dim=self.num_users, embeddings_regularizer=l2(self.emb_reg) )
        emb_item_mf = Embedding( output_dim=self.factors, input_dim=self.num_items, embeddings_regularizer=l2(self.emb_reg) )
        emb_item = Embedding( output_dim=self.factors, input_dim=self.num_items, embeddings_regularizer=l2(self.emb_reg) )
        
        if self.include_artist:
            emb_user_artist_mf = Embedding( output_dim=self.factors, input_dim=self.num_artists, embeddings_regularizer=l2(self.emb_reg) )
            emb_artist_mf = Embedding( output_dim=self.factors, input_dim=self.num_artists, embeddings_regularizer=l2(self.emb_reg) )
            emb_artist = Embedding( output_dim=self.factors, input_dim=self.num_artists, embeddings_regularizer=l2(self.emb_reg) )
                
        #MF PART     
          
        uemb = kl.Flatten()( emb_user_mf( user ) )
        iemb = kl.Flatten()( emb_item_mf( item ) )
                
        mf_dot = kl.Dot(1)( [uemb, iemb] )
        mf_mul = kl.Multiply()( [uemb, iemb] )
        
        mf_vector = kl.Concatenate()( [mf_mul, mf_dot] )
        
        #mf_vector = mf_mul
        
        if self.include_artist:
            uemb = kl.Flatten()( emb_user_artist_mf( user ) )
            aemb = kl.Flatten()( emb_artist_mf( item ) )
            mf_dot = kl.Dot(1)( [uemb, aemb] )
            mf_mul = kl.Multiply()( [uemb, aemb] )
            
            mf_vector = kl.Concatenate()( [mf_vector, mf_mul, mf_dot] )
        
        #MLP PART
        
        uemb = kl.Flatten()( emb_user( user ) )
        iemb = kl.Flatten()( emb_item( item ) )
        
        mlp_vector = kl.Concatenate()( [uemb, iemb] )
        if self.include_artist:
            emba = kl.Flatten()( emb_artist( artist ) )
            mlp_vector = kl.Concatenate()( [mlp_vector, emba] )
        
        for i in range( len(self.layers) ):
            layer = kl.Dense( self.layers[i], activation='relu', name="layer%d" %i, kernel_regularizer=l2(self.layer_reg) )
            mlp_vector = layer(mlp_vector)
        
        #PRED PART
        
        comb = kl.Concatenate()( [ mf_vector , mlp_vector ] ) #, uemb ] )
        
        fff = kl.Dense( 1, activation='linear', kernel_initializer='lecun_uniform', kernel_regularizer=l2(self.layer_reg) )
        res = fff(comb)
        
        inputs = [ user, item ] #+ [artist
        if self.include_artist:
            inputs += [ artist ]
        outputs = [ res ]
        
        predict_model = km.Model( inputs, outputs )
        
        current_user = kl.Input( ( 1, ), name="current_user" )# , batch_shape=(self.batch, self.steps) )
        current_item_pos = kl.Input( (1,), dtype=self.intX, name="current_item_pos" )#, batch_shape=(self.batch,1) )
        current_item_neg = kl.Input( (1,), dtype=self.intX, name="current_item_neg" )#, batch_shape=(self.batch,1) )
        
        pred_from_pos = [ current_user, current_item_pos ]
        pred_from_neg = [ current_user, current_item_neg ]
        
        if self.include_artist:
            current_artist_pos = kl.Input( ( 1, ), name="current_artist_pos" )# , batch_shape=(self.batch, self.steps) )
            current_artist_neg = kl.Input( ( 1, ), name="current_artist_neg" )# , batch_shape=(self.batch, self.steps) )
            pred_from_neg += [current_artist_neg]
            pred_from_pos += [current_artist_pos]
            
        current_res_pos = predict_model( pred_from_pos ) #, current_user ] )
        current_res_neg = predict_model( pred_from_neg ) #, current_user ] )
        
        inputs = [ current_user, current_item_pos, current_item_neg ] #+ [current_user]
        if self.include_artist:
            inputs += [current_artist_pos,current_artist_neg]
        outputs = [ current_res_pos, current_res_neg ]
        
        model = km.Model( inputs, outputs )
        model.add_loss(K.mean( self.bpr(outputs) ))
        
        if self.optimizer == 'adam': 
            opt = keras.optimizers.Adam(lr=self.learning_rate)
        elif self.optimizer == 'adagrad':
            opt = keras.optimizers.Adagrad(lr=self.learning_rate)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=self.learning_rate*10)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=self.learning_rate)
        
        model.compile( optimizer=opt )
                
        return model, predict_model
    
    def bpr(self, out):
        pos, neg = out
        obj = -K.sum( K.log( K.sigmoid( pos - neg ) ) )
        return obj
    
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
        
        if len(sitems) == 0:
            res_dict = {}
            res_dict['track_id'] =  []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)
        
                
        u = np.full( self.num_items , self.usermap[playlist_id], dtype=self.intX)
        i = np.array( self.itemmap.values )
        input = [ u,i ]
        if self.include_artist:
            a = np.array( self.artistmap[ self.itemartistmap[ self.itemmap.values ] ] )
            input += [a]
#         usera = np.zeros((1))
#         usera[0] = self.usermap[input_user_id]
        
        
        
        predictions = self.predict_model.predict( input, batch_size=len(i) ) #, usera ] )
        #predictions = self.predict( self.session_items, self.itemmap[input_item_id], self.usermap[input_user_id] )
        
        try:
        
            # Create things in the format
            res_dict = {}
            res_dict['track_id'] =  list(self.itemmap.index)
            res_dict['confidence'] = predictions.T[0]
            res = pd.DataFrame.from_dict(res_dict)
            
            res = res[ ~np.in1d( res.track_id, sitems ) ]
            
            res.sort_values( 'confidence', ascending=False, inplace=True )
            
        except Exception:
            print( 'hö' )
            print( self.itemmap.index )
            print( predictions )
            print( len(predictions[0]) )
            exit()
                
        return res.head(500)
        
