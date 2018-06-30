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


class CollaborativeMLP:
    '''
    CollaborativeMLP( factors=32, layers=[32,16,8], hidden_act='relu', final_act='sigmoid', batch=100, optimizer='adam', learning_rate=0.01, reg=0.01, layer_reg=0.01, emb_reg=0.01, dropout=0.0, epochs=10, include_artist=False, embeddings=None, folder=None,  order='random', session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key= 'artist_id', time_key= 'pos' )

    Parameters
    -----------
    '''
    
    def __init__( self, factors=32, layers=[32,16,8], hidden_act='relu', final_act='sigmoid', batch=100, optimizer='adam', learning_rate=0.01, reg=0.01, layer_reg=0.01, emb_reg=0.01, dropout=0.0, epochs=10, include_artist=False, embeddings=None, folder=None,  order='random', session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key= 'artist_id', time_key= 'pos' ):
       
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
        self.hidden_act = hidden_act
        self.final_act = final_act
        
        self.session_key = session_key
        self.item_key = item_key
        self.user_key = user_key
        self.time_key = time_key
        self.artist_key = artist_key
                
        self.emb_reg = emb_reg
        self.layer_reg = layer_reg
        self.final_reg = reg
        
        self.embeddings = embeddings
        self.folder = folder
        self.callbacks = []
                
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
        
        self.predict_model = self.init_model( train )
        
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
                
        start = time.time()
        
        for j in range( self.epochs ):
            
            starttmp = time.time()
            U, I, L, A = self.get_train_data( train )
            print( 'finished creating samples in {}'.format(  ( time.time() - starttmp ) ) )
            
            print( 'train epoch {} with {} examples'.format( j, len(U) ) )
            
            input = [np.array(U), np.array(I)]
            if self.include_artist:
                input += [np.array(A)]
            
            hist = self.predict_model.fit(input, #input
                         np.array(L), # labels 
                         batch_size=self.batch, epochs=1, shuffle=True, verbose=2 )
            
            print( 'finished epoch {} in {}s'.format( j, ( time.time() - start ) ) )
            
            for c in self.callbacks:
                if hasattr(c, 'callback'):
                    getattr(c,'callback')( self )
     
    def get_train_data( self, train, neg_samples=1 ):
             
        train = train.sample(frac=1).reset_index(drop=True)
        train['Label'] = 1
        
        trainn = train.copy()
        trainn['Label'] = 0
        trainn['ItemIdx'] = np.random.choice( self.itemmap.values, len(train) )
        
        
        if self.include_artist:
            trainn['ArtistIdx'] = self.itemartistmap[ trainn['ItemIdx'].values ].values
        else:
            trainn['ArtistIdx'] = 0
        
        trainn = pd.concat([train,trainn])
        if self.order == 'random':
            trainn = trainn.sample(frac=1).reset_index(drop=True)
        else:
            trainn = trainn.sort_values([self.user_key,self.item_key])
                        
        user = trainn['UserIdx'].values
        item = trainn['ItemIdx'].values
        label = trainn['Label'].values
        artist = trainn['ArtistIdx'].values
        
        return user, item, label, artist
    
    def get_latent( self, ids, type='als', size=64, col='track_id', count=False ):
                
        key = type + '_' + col + '_features' + ( '_cnt' if count else '' ) + '.' + str(size) + '.csv'
        
        features = pd.read_csv( self.folder + key )  
        features.index = features[col]
                     
        ft = features.ix[ids]
        LF = [ 'lf_'+str(i) for i in range(size) ]
        res = ft[LF].values
         
        return res
     
    def init_model(self, train, std=0.01):
        
        #current_item = kl.Input( ( 1, ), name="current_item" )
        
        item = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.,self.steps) )
        user = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        if self.include_artist:
            artist = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        trainable = True
        if self.embeddings == 'fixed':
            trainable = False
        
        emb_user = Embedding( embeddings_initializer='random_normal', output_dim=self.factors, input_dim=self.num_users, embeddings_regularizer=l2(self.emb_reg), trainable=trainable )
        emb_item = Embedding( embeddings_initializer='random_normal', output_dim=self.factors, input_dim=self.num_items, embeddings_regularizer=l2(self.emb_reg), trainable=trainable )
        
        if self.embeddings != None:
            userw = self.get_latent( self.usermap.index, size=self.factors, col='playlist_id' )
            itemw = self.get_latent( self.itemmap.index, size=self.factors )
            emb_user.build((None,))
            emb_item.build((None,))
            emb_user.set_weights([userw])
            emb_item.set_weights([itemw])
        
        if self.include_artist:
            emb_artist = Embedding( output_dim=self.factors, input_dim=self.num_artists, embeddings_regularizer=l2(self.emb_reg) )
                
        #MLP PART
        
        uemb = kl.Flatten()( emb_user( user ) )
        iemb = kl.Flatten()( emb_item( item ) )
        
        mlp_vector = kl.Concatenate()( [uemb, iemb] )
        if self.include_artist:
            emba = kl.Flatten()( emb_artist( artist ) )
            mlp_vector = kl.Concatenate()( [mlp_vector, emba] )
        
        for i in range( len(self.layers) ):
            layer = kl.Dense( self.layers[i], activation=self.hidden_act, name="layer%d" %i, kernel_regularizer=l2(self.layer_reg) )
            #bn = kl.BatchNormalization()
            #act = kl.Activation('relu')
            #mlp_vector = act( bn( layer(mlp_vector) ) )
            mlp_vector = layer(mlp_vector)
        
        #PRED PART
        
        fff = kl.Dense( 1, activation=self.final_act, kernel_initializer='lecun_uniform', kernel_regularizer=l2(self.final_reg) )
        res = fff( mlp_vector )
        
        inputs = [ user, item ] #+ [artist]
        if self.include_artist:
            inputs += [artist]
        outputs = [ res ]
        
        model = km.Model( inputs, outputs )
        
        if self.optimizer == 'adam': 
            opt = keras.optimizers.Adam(lr=self.learning_rate)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=self.learning_rate)
        elif self.optimizer == 'adamax':
            opt = keras.optimizers.Adamax(lr=self.learning_rate)
        elif self.optimizer == 'adagrad':
            opt = keras.optimizers.Adagrad(lr=self.learning_rate)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=self.learning_rate)
        
        model.compile( optimizer=opt, loss='binary_crossentropy' )
        plot_model( model, to_file='mlp.png' )
        
        return model
    
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
    
    def add_epoch_callback(self, clazz):
        
        self.callbacks.append(clazz)
        
