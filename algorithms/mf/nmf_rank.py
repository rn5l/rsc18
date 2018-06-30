from _operator import itemgetter
from math import sqrt
import random
import time

from pympler import asizeof
import numpy as np
import pandas as pd
from math import log10
import scipy.sparse
from scipy.sparse.csc import csc_matrix
import theano
import theano.tensor as T
import keras.layers as kl
import keras.models as km
import keras.backend as K
from datetime import datetime as dt
from datetime import timedelta as td
from keras.layers.embeddings import Embedding
import keras
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model


class RankNetNeuralMF:
    '''
    RankNetNeuralMF( factors=16, layers=[64,32,16,8], batch=100, optimizer='adam', learning_rate=0.01, reg=0.01, emb_reg=0.01, dropout=0.0, epochs=10, add_dot=False, include_artist=False, order='random', session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key= 'artist_id', time_key= 'pos' )

    Parameters
    -----------
    '''
    
    def __init__( self, factors=16, layers=[64,32,16,8], batch=100, optimizer='adam', learning_rate=0.01, reg=0.01, emb_reg=0.01, dropout=0.0, epochs=10, add_dot=False, include_artist=False, order='random', session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key= 'artist_id', time_key= 'pos' ):
       
        self.factors = factors
        self.layers = layers
        self.batch = batch
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.regularization = reg        
        self.dropout = dropout
        self.epochs = epochs
        self.order = order
        self.include_artist = include_artist
        self.add_dot = add_dot
        
        self.session_key = session_key
        self.item_key = item_key
        self.user_key = user_key
        self.time_key = time_key
        self.artist_key = artist_key
                
        self.emb_reg = emb_reg
        self.final_reg = reg
                
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
        
        model, predict_model = self.init_model( train )
        
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
                
        start = time.time()
        
        for j in range( self.epochs ):
            
            starttmp = time.time()
            U, I, IN, A, AN = self.get_train_data( train )
            print( 'finished creating samples in {}'.format(  ( time.time() - starttmp ) ) )
            
            print( 'train epoch {} with {} examples'.format( j, len(U) ) )
            
            input = [U, I, IN]
            if self.include_artist:
                input += [A,AN]
            
            L = np.ones(len(U))
            
            hist = model.fit(input, #input
                         L, # labels => all one
                         batch_size=self.batch, epochs=1, shuffle=True, verbose=2 )
            
            print( 'finished epoch {} in {}s'.format( j, ( time.time() - start ) ) )
    
        
        self.predict_model = predict_model
        
    def get_train_data( self, train ):
             
        #train = train.sample(frac=1).reset_index(drop=True)
        
        train['ItemIdxNeg'] = np.random.choice( self.itemmap.values, len(train) )
        items = train['ItemIdxNeg'].values
        train['ArtistIdxNeg'] = self.itemartistmap[ items ].values
        
        return train['UserIdx'].values, train['ItemIdx'].values, train['ItemIdxNeg'].values, train['ArtistIdx'].values, train['ArtistIdxNeg'].values
      
    def init_model(self, train, std=0.01):
        
        predict = self.get_pmodel()
        
        user = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        item = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.,self.steps) )
        item_neg = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.,self.steps) )
        
        if self.include_artist:
            artist = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
            artist_neg = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        inputs_pos = [ user, item ] #+ [artist]
        inputs_neg = [ user, item_neg ] #+ [artist]
        if self.include_artist:
            inputs_pos += [artist]
            inputs_neg += [artist_neg]
            
        res = predict( inputs_pos )
        res_neg = predict( inputs_neg )
        
        diff = kl.Subtract()( [res,res_neg] )
        diff = kl.Activation( 'sigmoid' )( diff )
        
        inputs = [ user, item, item_neg ] #+ [artist]
        if self.include_artist:
            inputs += [artist, artist_neg]
        outputs = [ diff ]
        
        model = km.Model( inputs, outputs )
        
        if self.optimizer == 'adam': 
            opt = keras.optimizers.Adam(lr=self.learning_rate)
        elif self.optimizer == 'adagrad':
            opt = keras.optimizers.Adagrad(lr=self.learning_rate)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=self.learning_rate)
        elif self.optimizer == 'adamax':
            opt = keras.optimizers.Adamax(lr=self.learning_rate)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=self.learning_rate*10)
        
        model.compile( optimizer=opt, loss='binary_crossentropy' )
        plot_model( model, to_file='nmf_rank.png' )
        
        inputs = [ user, item ]
        if self.include_artist:
            inputs += [artist]
                
        return model, predict
    
    def get_pmodel(self):
        
        user = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        item = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.,self.steps) )
        
        if self.include_artist:
            artist = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        self.emb_user_mf = Embedding( output_dim=self.factors, input_dim=self.num_users, embeddings_regularizer=l2(self.emb_reg) )
        self.emb_item_mf = Embedding( output_dim=self.factors, input_dim=self.num_items, embeddings_regularizer=l2(self.emb_reg) )
        
        if self.include_artist:
            self.emb_user_artist_mf = Embedding( output_dim=self.factors, input_dim=self.num_users, embeddings_regularizer=l2(self.emb_reg) )
            self.emb_artist_mf = Embedding( output_dim=self.factors, input_dim=self.num_artists, embeddings_regularizer=l2(self.emb_reg) )
        
        self.fff = kl.Dense( 1, activation='sigmoid', kernel_initializer='lecun_uniform', bias_regularizer=l2(self.final_reg) )
                
        if self.include_artist:  
            res = self.get_score(user, item, artist)
        else:
            res = self.get_score(user, item)
        
        inputs = [ user, item ]
        if self.include_artist:
            inputs += [artist]
        outputs = [ res ]
        
        predict_model = km.Model( inputs, outputs )
        
        return predict_model
        
    
    def get_score(self,user,item, artist=None):
        
        uemb = kl.Flatten()( self.emb_user_mf( user ) )
        iemb = kl.Flatten()( self.emb_item_mf( item ) )
                
        mf_vector = kl.Multiply()( [uemb, iemb] )
        if self.add_dot:
            mf_dot = kl.Dot(1)( [uemb, iemb] )
            mf_vector = kl.Concatenate()( [mf_vector, mf_dot] )
            
        if self.include_artist:
            uemb = kl.Flatten()( self.emb_user_artist_mf( user ) )
            aemb = kl.Flatten()( self.emb_artist_mf( artist ) )
            mf_mul = kl.Multiply()( [uemb, aemb] )
            if self.add_dot:
                mf_dot = kl.Dot(1)( [uemb, aemb] )
                mf_mul = kl.Concatenate()( [mf_mul, mf_dot] )
            
            mf_vector = kl.Concatenate()( [mf_vector, mf_mul] )
                
        res = self.fff(mf_vector)
        
        return res
        
    
    def predict( self, name=None, tracks=None, playlist_id=None, artists=None, num_hidden=None ):
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
        i = self.itemmap.values
        input = [ u, i ]
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
        
