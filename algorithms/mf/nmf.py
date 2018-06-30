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
from keras.layers.normalization import BatchNormalization


class NeuralMF:
    '''
    NeuralMF( factors=16, layers=[64,32,16,8], batch=100, optimizer='adam', learning_rate=0.01, reg=0.01, emb_reg=0.01, dropout=0.0, epochs=10, neg_samples=1, add_dot=False, include_artist=False, batch_normalization=False, embeddings=None, folder=None, filter=None, order='random', sampling='fast', session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key= 'artist_id', time_key= 'pos' )

    Parameters
    -----------
    '''
    
    def __init__( self, factors=16, layers=[64,32,16,8], batch=100, optimizer='adam', learning_rate=0.01, reg=0.01, emb_reg=0.01, dropout=0.0, epochs=10, neg_samples=1, add_dot=False, include_artist=False, batch_normalization=False, embeddings=None, folder=None, filter=None, order='random', sampling='fast', session_key = 'playlist_id', item_key= 'track_id', user_key= 'playlist_id', artist_key= 'artist_id', time_key= 'pos' ):       
        
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
        self.neg_samples = neg_samples
        self.add_dot = add_dot
        self.batch_normalization = batch_normalization
        self.sampling = sampling
        
        self.session_key = session_key
        self.item_key = item_key
        self.user_key = user_key
        self.time_key = time_key
        self.artist_key = artist_key
                
        self.emb_reg = emb_reg
        self.final_reg = reg

        self.filter = filter
        self.callbacks = []
        self.embeddings = embeddings
        self.folder = folder
                        
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
                
        if self.filter is not None:
            data = self.filter_data(data, min_uc=self.filter[0], min_sc=self.filter[1])
        
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
        
        self.itemusermap = train.groupby( 'ItemIdx' )['UserIdx'].apply( set )
        self.itemusermap = pd.Series( index=self.itemusermap.index, data = self.itemusermap.values )
        
        self.predict_model = self.init_model( train )
        
        print( 'finished init model in {}'.format(  ( time.time() - start ) ) )
                
        start = time.time()
        
        for j in range( self.epochs ):
            
            starttmp = time.time()
            U, I, L, A = self.get_train_data( train, neg_samples=self.neg_samples )
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
        
        trainf = train
        
        def random_choice( user ):
            search = True
            while search:
                item = np.random.choice( self.itemmap.values )
                search = user in self.itemusermap[item]
            return item
        
        for i in range( neg_samples ):
            trainn = pd.DataFrame()
            trainn['UserIdx'] = train['UserIdx']
            trainn['Label'] = 0
            if self.sampling == 'fast':
                trainn['ItemIdx'] = np.random.choice( self.itemmap.values, len(train) )
            else:
                trainn['ItemIdx'] = trainn['UserIdx'].apply( random_choice )
            
            if self.include_artist:
                trainn['ArtistIdx'] = self.itemartistmap[ trainn['ItemIdx'].values ].values
            else:
                trainn['ArtistIdx'] = 0
                
            trainf = pd.concat([trainf,trainn])
            
        if self.order == 'random':
            trainf = trainf.sample(frac=1).reset_index(drop=True)
        else:
            trainf = trainf.sort_values([self.user_key,self.item_key])
        
        user = trainf['UserIdx'].values
        item = trainf['ItemIdx'].values
        label = trainf['Label'].values
        artist = trainf['ArtistIdx'].values
        
        return user, item, label, artist
       
    def init_model(self, train, std=0.01):
        
        #current_item = kl.Input( ( 1, ), name="current_item" )
        
        item = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.,self.steps) )
        user = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        if self.include_artist:
            artist = kl.Input( (1,), dtype=self.intX )#, batch_shape=(self.batch,1) )
        
        trainable = True
        if self.embeddings == 'fixed':
            trainable = False
        
        emb_user_mf = Embedding( output_dim=self.factors, input_dim=self.num_users, embeddings_regularizer=l2(self.emb_reg), trainable=trainable )
        emb_item_mf = Embedding( output_dim=self.factors, input_dim=self.num_items, embeddings_regularizer=l2(self.emb_reg), trainable=trainable )
        
        if self.embeddings != None:
            userw = self.get_latent( self.usermap.index, size=self.factors, col='playlist_id' )
            itemw = self.get_latent( self.itemmap.index, size=self.factors )
            emb_user_mf.build((None,))
            emb_item_mf.build((None,))
            emb_user_mf.set_weights([userw])
            emb_item_mf.set_weights([itemw])
                
        if self.include_artist:
            emb_user_artist_mf = Embedding( output_dim=self.factors, input_dim=self.num_users, embeddings_regularizer=l2(self.emb_reg) )
            emb_artist_mf = Embedding( output_dim=self.factors, input_dim=self.num_artists, embeddings_regularizer=l2(self.emb_reg) )
                
        #MF PART
                
        uemb = kl.Flatten()( emb_user_mf( user ) )
        iemb = kl.Flatten()( emb_item_mf( item ) )
                
        mf_vector = kl.Multiply()( [uemb, iemb] )
        if self.add_dot:
            mf_dot = kl.Dot(1)( [uemb, iemb] )
            mf_vector = kl.Concatenate()( [mf_vector, mf_dot] )
            
        if self.include_artist:
            uemb = kl.Flatten()( emb_user_artist_mf( user ) )
            aemb = kl.Flatten()( emb_artist_mf( artist ) )
            mf_mul = kl.Multiply()( [uemb, aemb] )
            if self.add_dot:
                mf_dot = kl.Dot(1)( [uemb, aemb] )
                mf_mul = kl.Concatenate()( [mf_mul, mf_dot] )
            
            mf_vector = kl.Concatenate()( [mf_vector, mf_mul] ) #, mf_dot] )
        
        #PRED PART
        
        if self.batch_normalization:
            fff = kl.Dense( 1, kernel_initializer='lecun_uniform', bias_regularizer=l2(self.final_reg) )
            bn = kl.BatchNormalization()
            act = kl.Activation( 'sigmoid' )
            res = act( bn( fff(mf_vector) ) )
        else:
            fff = kl.Dense( 1, activation='sigmoid', kernel_initializer='lecun_uniform', bias_regularizer=l2(self.final_reg) )
            res = fff(mf_vector)
        
        inputs = [ user, item ] #+ [artist]
        if self.include_artist:
            inputs += [artist]
        outputs = [ res ]
        
        model = km.Model( inputs, outputs )
        
        if self.optimizer == 'adam': 
            opt = keras.optimizers.Adam(lr=self.learning_rate)
        elif self.optimizer == 'adamax':
            opt = keras.optimizers.Adamax(lr=self.learning_rate)
        elif self.optimizer == 'nadam':
            opt = keras.optimizers.Nadam(lr=self.learning_rate)
        elif self.optimizer == 'adagrad':
            opt = keras.optimizers.Adagrad(lr=self.learning_rate)
        elif self.optimizer == 'adadelta':
            opt = keras.optimizers.Adadelta(lr=self.learning_rate)
        elif self.optimizer == 'sgd':
            opt = keras.optimizers.SGD(lr=self.learning_rate)
        
        model.compile( optimizer=opt, loss='binary_crossentropy' )
        plot_model( model, to_file='nmf.png' )
        
        return model
    
    def get_latent( self, ids, type='als', size=64, col='track_id', count=False ):
                
        key = type + '_' + col + '_features' + ( '_cnt' if count else '' ) + '.' + str(size) + '.csv'
        
        features = pd.read_csv( self.folder + key )  
        features.index = features[col]
                     
        ft = features.ix[ids]
        LF = [ 'lf_'+str(i) for i in range(size) ]
        res = ft[LF].values
         
        return res
    
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
        
        sitems = np.array(tracks) if tracks is not None else np.array([])
        sitems = sitems[np.isin(sitems, self.itemmap.index)]
        
        if len(sitems) == 0:
            res_dict = {}
            res_dict['track_id'] = []
            res_dict['confidence'] = []
            return pd.DataFrame.from_dict(res_dict)   
        
        u = np.full( self.num_items , self.usermap[playlist_id], dtype=self.intX)
        i = np.array( self.itemmap.values )
        input = [ u,i ]
        if self.include_artist:
            a = np.array( self.itemartistmap[ self.itemmap.values ] )
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
     
