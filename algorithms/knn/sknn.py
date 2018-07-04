from _operator import itemgetter
import gc
from math import sqrt, log10
import math
import os
import pickle
import random
import time

import psutil

from nltk import tokenize as tokenise, stem
import numpy as np
import pandas as pd


class SessionKNN: 
    '''
    SessionKNN(k, sample_size=1000, sampling='recent', similarity='cosine', title_boost=0, seq_weighting=None, idf_weight=None, pop_weight=False, pop_boost=0, artist_boost=0, remind=False, sim_cap=0, normalize=True, neighbor_decay=0, session_key = 'playlist_id', item_key= 'track_id', time_key= 'pos', folder=None, return_num_preds=500 )

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__( self, k, sample_size=1000, sampling='recent', similarity='cosine', title_boost=0, seq_weighting=None, idf_weight=None, pop_weight=False, pop_boost=0, artist_boost=0, remind=False, sim_cap=0, normalize=True, neighbor_decay=0, session_key = 'playlist_id', item_key= 'track_id', time_key= 'pos', folder=None, return_num_preds=500 ):
       
        self.k = k
        self.sample_size = sample_size
        self.sampling = sampling
        self.similarity = similarity
        self.pop_boost = pop_boost
        self.artist_boost = artist_boost
        self.title_boost = title_boost
        self.seq_weighting = seq_weighting
        self.idf_weight = idf_weight
        self.pop_weight = pop_weight
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.remind = remind
        self.normalize = normalize
        self.sim_cap = sim_cap
        self.neighbor_decay = neighbor_decay
        self.return_num_preds = return_num_preds
        
        #updated while recommending
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        # cache relations once at startup
        self.session_item_map = dict() 
        self.item_session_map = dict()
        self.session_time = dict()
        self.folder = folder
        
        self.sim_time = 0
        
    def train( self, data, test=None ):
        '''
        Trains the predictor.
        
        Parameters
        --------
        data: pandas.DataFrame
            Training data. It contains the transactions of the sessions. It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
            It must have a header. Column names are arbitrary, but must correspond to the ones you set during the initialization of the network (session_key, item_key, time_key properties).
            
        '''
        
        train = data['actions']
        playlists = data['playlists']
        
        folder = self.folder
        
        if folder is not None and os.path.isfile( folder + 'session_item_map.pkl' ):
            self.session_item_map = pickle.load( open( folder + 'session_item_map.pkl', 'rb') )
            self.session_time = pickle.load( open( folder + 'session_time.pkl', 'rb' ) )
            self.item_session_map = pickle.load( open( folder + 'item_session_map.pkl', 'rb' ) )
        else: 
            
            index_session = train.columns.get_loc( self.session_key )
            index_item = train.columns.get_loc( self.item_key )
            #index_time = train.columns.get_loc( self.time_key )
            
            session = -1
            session_items = set()
            timestamp = -1
            
            cnt = 0
            tstart = time.time()
            
            timemap = pd.Series( index=playlists.playlist_id, data=playlists.modified_at )
            
            for row in train.itertuples(index=False):
                # cache items of sessions
                if row[index_session] != session:
                    if len(session_items) > 0:
                        self.session_item_map.update({session : session_items})
                        # cache the last time stamp of the session
                        self.session_time.update({session : timestamp})
                    session = row[index_session]
                    session_items = set()
                timestamp = timemap[row[index_session]]
                session_items.add(row[index_item])
                
                # cache sessions involving an item
                map_is = self.item_session_map.get( row[index_item] )
                if map_is is None:
                    map_is = set()
                    self.item_session_map.update({row[index_item] : map_is})
                map_is.add(row[index_session])
                
                cnt += 1
                
                if cnt % 100000 == 0:
                    print( ' -- finished {} of {} rows in {}s'.format( cnt, len(train), (time.time() - tstart) ) )
                
            # Add the last tuple    
            self.session_item_map.update({session : session_items})
            self.session_time.update({session : timestamp})
            
            if folder is not None:
                pickle.dump( self.session_item_map,  open( folder + 'session_item_map.pkl', 'wb' ) )
                pickle.dump( self.session_time, open( folder + 'session_time.pkl', 'wb' ) )
                pickle.dump( self.item_session_map, open( folder + 'item_session_map.pkl', 'wb' ) )
        
        self.item_pop = pd.DataFrame()
        self.item_pop['pop'] = train.groupby( self.item_key ).size()
        #self.item_pop['pop'] = self.item_pop['pop'] / self.item_pop['pop'].max()
        self.item_pop['pop'] = self.item_pop['pop'] / len( train )
        self.item_pop = self.item_pop['pop'].to_dict()
        
        if self.idf_weight != None:
            self.idf = pd.DataFrame()
            self.idf['idf'] = train.groupby( self.item_key ).size()
            self.idf['idf'] = np.log( train[self.session_key].nunique() / self.idf['idf'] )
            self.idf = self.idf['idf'].to_dict()
        
        if self.title_boost > 0:
            self.stemmer = stem.PorterStemmer()
            playlists['name'] = playlists['name'].apply(lambda x: self.normalise(str(x), True, True ))
            playlists['name_id'] = playlists['name'].astype( 'category' ).cat.codes
            self.namemap = playlists.groupby('name').name_id.min()
            self.namemap = pd.Series( index=self.namemap.index, data=self.namemap.values )
            self.plnamemap = pd.Series( index=playlists['playlist_id'].values, data=playlists['name_id'].values )
            
        self.artist_map = pd.DataFrame()
        self.artist_map['artist_id'] = train.groupby( self.item_key ).artist_id.min()
        self.artist_map = self.artist_map['artist_id'].to_dict()
                
        self.tall = 0
        self.tneighbors = 0
        self.tscore = 0
        self.tartist = 0
        self.tformat = 0
        self.tsort = 0
        self.tnorm = 0
        self.thead = 0
        self.count = 0
                
    def predict( self, plname, tracks, playlist_id=None, artists=None, num_hidden=None ):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.
                
        Parameters
        --------
        plnam : string
            The session IDs of the event.
        tracks : list of int
            The item ID of the event. Must be in the set of item IDs of the training set.
            
        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.
        
        '''
        
        
        tstart = time.time()
        tstartall = tstart
        #items = tracks.track_id.values    
        items = tracks if tracks is not None else []
        
        if len(items) == 0: 
            res_dict = {}
            res_dict['track_id'] =  []
            res_dict['confidence'] = []
            res = pd.DataFrame.from_dict(res_dict)
            return res
                
        iset = set(items)
        neighbors = self.find_neighbors( iset, plname )
        self.tneighbors += (time.time() - tstart)
        
        tstart = time.time()
        scores, simsum, count = self.score_items( neighbors, items, iset )
        self.tscore += (time.time() - tstart)
        
        #push popular ones
        if self.pop_boost > 0:
                
            for key in scores:
                item_pop = self.item_pop[key]
                # Gives some minimal MRR boost?
                scores.update({key : (scores[key] + ( self.pop_boost * item_pop) )})
        
        tstart = time.time()
        
        if self.artist_boost > 0 and artists is not None:
            
            y = np.bincount(artists)
            ii = np.nonzero(y)[0]
            freq = pd.Series( index=ii, data=y[ii] )
            
#             res['artist_id'] = res.track_id.map( lambda t : self.artist_map[t] )
#             res = res.merge( freq.to_frame('tmp'), how="left", left_on='artist_id', right_index=True )
#             res['confidence'] += res['tmp'].fillna(0) * self.artist_boost
#             del res['tmp']
            
            for key in scores:
                if self.artist_map[key] in freq.index:
                    factor = freq[self.artist_map[key]] / len(freq)
                else:
                    factor = 0
                      
                scores.update({key : (scores[key] + ( scores[key] * self.artist_boost * factor) )})
                
        self.tartist += (time.time() - tstart)
        
        tstart = time.time()
        # Create things in the format
        res_dict = {}
        res_dict['track_id'] = [x for x in scores]
        res_dict['confidence'] = [scores[x] for x in scores]
        if self.neighbor_decay > 0:
            res_dict['count'] = [count[x] for x in scores]
        res = pd.DataFrame.from_dict(res_dict)
        if self.neighbor_decay > 0:
            res['count'] = ( res['count'] - res['count'].min() ) / ( res['count'].max() - res['count'].min() )
            res['confidence'] = res['confidence'] - ( res['confidence'] * res['count'] * self.neighbor_decay)
        self.tformat += (time.time() - tstart)
             
        tstart = time.time()
        if self.normalize:
            res['confidence'] = res['confidence'] / simsum
            #res['confidence'] = ( res['confidence']  - res['confidence'].min() )/ ( res['confidence'].max() - res['confidence'].min() )
        self.tnorm += (time.time() - tstart)
        
        tstart = time.time()
        res.sort_values( ['confidence','track_id'], ascending=[False,True], inplace=True )
        res = res.reset_index(drop=True)
        self.tsort += (time.time() - tstart)

        tstart = time.time()
        res = res.head(self.return_num_preds)
        self.thead += (time.time() - tstart)
        
        self.tall += (time.time() - tstartall)
        
        self.count += 1
        
        #rlist = [ (self.tall / self.count), (self.tneighbors / self.count), (self.tscore / self.count), (self.tartist / self.count), (self.tsort / self.count), (self.tnorm / self.count) ]
        #print( ','.join( map( str, rlist ) ) )
                
        return res


    def jaccard(self, first, second):
        '''
        Calculates the jaccard index for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        sc = time.clock()
        intersection = len(first & second)
        union = len(first | second )
        res = intersection / union
        
        self.sim_time += (time.clock() - sc)
        
        return res 
    
    def cosine(self, first, second):
        '''
        Calculates the cosine similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( sqrt(la) * sqrt(lb) )

        return result
    
    def tanimoto(self, first, second):
        '''
        Calculates the cosine tanimoto similarity for two sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        li = len(first&second)
        la = len(first)
        lb = len(second)
        result = li / ( la + lb -li )

        return result
    
    def binary(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        a = len(first&second)
        b = len(first)
        c = len(second)
        
        result = (2 * a) / ((2 * a) + b + c)

        return result
    
    def random(self, first, second):
        '''
        Calculates the ? for 2 sessions
        
        Parameters
        --------
        first: Id of a session
        second: Id of a session
        
        Returns 
        --------
        out : float value           
        '''
        return random.random()
    

    def items_for_session(self, session):
        '''
        Returns all items in the session
        
        Parameters
        --------
        session: Id of a session
        
        Returns 
        --------
        out : set           
        '''
        return self.session_item_map.get(session);
    
    
    def sessions_for_item(self, item_id):
        '''
        Returns all session for an item
        
        Parameters
        --------
        item: Id of the item session
        
        Returns 
        --------
        out : set           
        '''
        return self.item_session_map.get( item_id )
        
        
    def most_recent_sessions( self, sessions, number ):
        '''
        Find the most recent sessions in the given set
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        sample = set()

        tuples = list()
        for session in sessions:
            time = self.session_time.get( session )
            if time is None:
                print(' EMPTY TIMESTAMP!! ', session)
            tuples.append((session, time))
            
        tuples = sorted(tuples, key=itemgetter(1), reverse=True)
        #print 'sorted list ', sortedList
        cnt = 0
        for element in tuples:
            cnt = cnt + 1
            if cnt > number:
                break
            sample.add( element[0] )
        #print 'returning sample of size ', len(sample)
        return sample
        
        
    def possible_neighbor_sessions(self, session_items):
        '''
        Find a set of session to later on find neighbors in.
        A self.sample_size of 0 uses all sessions in which any item of the current session appears.
        self.sampling can be performed with the options "recent" or "random".
        "recent" selects the self.sample_size most recent sessions while "random" just choses randomly. 
        
        Parameters
        --------
        sessions: set of session ids
        
        Returns 
        --------
        out : set           
        '''
        
        relevant_sessions = set()
        for item in session_items:
            relevant_sessions = relevant_sessions | self.sessions_for_item( item );
               
        if self.sample_size == 0: #use all session as possible neighbors
            
            #print('!!!!! runnig KNN without a sample size (check config)')
            return relevant_sessions

        else: #sample some sessions
                                         
            if len(relevant_sessions) > self.sample_size:
                
                if self.sampling == 'recent':
                    sample = self.most_recent_sessions( relevant_sessions, self.sample_size )
                elif self.sampling == 'random':
                    sample = random.sample( relevant_sessions, self.sample_size )
                else:
                    sample = relevant_sessions[:self.sample_size]
                    
                return sample
            else: 
                return relevant_sessions
                        

    def calc_similarity(self, session_items, sessions, playlist_name=None ):
        '''
        Calculates the configured similarity for the items in session_items and each session in sessions.
        
        Parameters
        --------
        session_items: set of item ids
        sessions: list of session ids
        
        Returns 
        --------
        out : list of tuple (session_id,similarity)           
        '''
        
        #print 'nb of sessions to test ', len(sessionsToTest), ' metric: ', self.metric
        neighbors = []
        cnt = 0
        
        threshold = 0
        if self.sim_cap > 0:
            avg_size = np.mean( [len(self.items_for_session( session )) for session in sessions] )
            la = len(session_items)
            lb = avg_size
            li = min( math.ceil(lb*self.sim_cap), math.ceil(la*self.sim_cap) )
            threshold = li / ( sqrt(la) * sqrt(lb) )
            
        for session in sessions:
            cnt = cnt + 1
            # get items of the session, look up the cache first 
            session_items_test = self.items_for_session( session )
            
            similarity = getattr(self , self.similarity)(session_items_test, session_items)
            
            if self.title_boost > 0:
                if playlist_name in self.namemap and self.namemap[playlist_name] == self.plnamemap[session]:
                    similarity += similarity * self.title_boost
            
            if similarity > threshold:
                neighbors.append((session, similarity))
                
        return neighbors


    #-----------------
    # Find a set of neighbors, returns a list of tuples (sessionid: similarity) 
    #-----------------
    def find_neighbors( self, session_items, plname ):
        '''
        Finds the k nearest neighbors for the given session_id and the current item input_item_id. 
        
        Parameters
        --------
        session_items: set of item ids
        input_item_id: int 
        session_id: int
        
        Returns 
        --------
        out : list of tuple (session_id, similarity)           
        '''
        possible_neighbors = self.possible_neighbor_sessions( session_items )
        possible_neighbors = self.calc_similarity( session_items, possible_neighbors, plname )
        
        possible_neighbors = sorted( possible_neighbors, reverse=True, key=lambda x: x[1] )
        possible_neighbors = possible_neighbors[:self.k]
        
        return possible_neighbors
    
            
    def score_items(self, neighbors, current_session, iset):
        '''
        Compute a set of scores for all items given a set of neighbors.
        
        Parameters
        --------
        neighbors: set of session ids
        
        Returns 
        --------
        out : list of tuple (item, score)           
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        sum = 0
        count = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )
            step = 1
            
            if self.seq_weighting is not None: 
                for item in reversed( current_session ):
                    if item in items:
                        decay = getattr(self, self.seq_weighting)( step )
                        break
                    step += 1
            
            for item in items:
                
                if not self.remind and item in iset:
                    continue
                
                old_score = scores.get( item )
                new_score = session[1]
                
                if old_score is None:
                    new_score = new_score if not self.idf_weight else new_score + ( new_score * self.idf[item] * self.idf_weight )
                    new_score = new_score if not self.pop_weight else new_score / self.item_pop[item]
                    new_score = new_score if self.seq_weighting is None else new_score * decay
                else: 
                    new_score = new_score if not self.idf_weight else new_score + ( new_score * self.idf[item] * self.idf_weight )
                    new_score = new_score if not self.pop_weight else new_score / self.item_pop[item]
                    new_score = new_score if self.seq_weighting is None else new_score * decay
                    new_score = old_score + new_score
                    
                scores.update({item : new_score})
                
                if item not in count:
                    count[item] = 0
                count[item] += 1
                 
            sum += session[1]
                     
        return scores, sum, count
    
    def linear(self, i):
        return 1 - (0.1*i) if i <= 100 else 0
    
    def same(self, i):
        return 1
    
    def div(self, i):
        return 1/i
    
    def log(self, i):
        return 1/(log10(i+1.7))
    
    def quadratic(self, i):
        return 1/(i*i)
    
    def normalise(self, s, tokenize=True, stemm=True):
        if tokenize:
            words = tokenise.wordpunct_tokenize(s.lower().strip())
        else:
            words = s.lower().strip().split( ' ' )
        if stemm:
            return ' '.join([self.stemmer.stem(w) for w in words])
        else:
            return ' '.join([w for w in words])
        