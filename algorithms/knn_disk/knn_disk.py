import csv
import datetime
from math import log10, sqrt
import os
import time

from algorithms.baseline.Solution import Solution


class KNNDisk:

    def __init__( self, num_neighbours=500, tf_method='ratio', idf_method='log10', similarity='jaccard', sim_denom_add = 1, session_key = 'playlist_id', item_key= 'track_id', time_key= 'pos', folder=None, write_num_recs=500 ):
        self.folder=folder
        self.sim_denom_add=sim_denom_add
        self.similarity=similarity
        self.num_neighbours=num_neighbours
        self.write_num_recs=write_num_recs
        self.idf_method=idf_method
        self.tf_method=tf_method

        algo_name = 'knn_disk'
        config_string = 'tf-' + self.tf_method + '_idf-' + self.idf_method
        self.train_tfidf_file_path=self.folder + algo_name + '_tfidf-train_' + config_string + '.csv'
        self.test_tfidf_file_path=self.folder + algo_name + '_tfidf-test_' + config_string + '.csv'
        config_string = config_string + '_' + self.similarity + '-' + 's' + str(self.sim_denom_add)
        self.simfile_path=self.folder + algo_name + '_sim_' + config_string + '.csv'
        config_string = config_string + '_k' + str(self.num_neighbours)
        self.res_name=algo_name + '_' + config_string + '.csv'
        self.predsfile_path=self.folder + 'results' + str(self.write_num_recs) + '_' + self.res_name
        self.session_key = session_key
        self.item_key = item_key
    
    def train(self,train,test):
        self.run(train, test)
        self.sol = Solution( self.predsfile_path )
        self.sol.train(train,test)
    
    def predict( self, plname, tracks, playlist_id=None, artists=None, num_hidden=None ):
        return self.sol.predict( plname, tracks, playlist_id=playlist_id, artists=artists, num_hidden=num_hidden )
    
    def run(self, train, test):
        train_actions = train['actions']
        test_actions = test['actions']

        print(str(datetime.datetime.now()),'############################ start knn_disk run()')
        tstart = time.time()

        container_idx = test_actions.columns.get_loc( self.session_key )
        item_idx = test_actions.columns.get_loc( self.item_key )

        overwrite = False

        # calc tfidfs
        if not (os.path.isfile(self.train_tfidf_file_path) and os.path.isfile(self.test_tfidf_file_path)) or overwrite:
            print(str(datetime.datetime.now()),'start tfidf calc')
            train_container_items_list = self.data_as_dict(train_actions, container_idx, item_idx)
            test_container_items_list = self.data_as_dict(test_actions, container_idx, item_idx)
            train_tfidfs, test_tfidfs = self.calc_train_test_tfidfs(train_container_items_list, test_container_items_list)
            self.write_container_item_score(train_tfidfs, self.train_tfidf_file_path, ['playlist_id','track_id','tfidf'])
            self.write_container_item_score(test_tfidfs, self.test_tfidf_file_path, ['playlist_id','track_id','tfidf'])
        else:
            print(str(datetime.datetime.now()),'read existing tfidf file')
            train_tfidfs = self.read_container_item_score(self.train_tfidf_file_path, True)
            test_tfidfs = self.read_container_item_score(self.test_tfidf_file_path, True)

        # calc sim
        if not os.path.isfile( self.simfile_path ) or overwrite:
            print(str(datetime.datetime.now()),'start similarity calc')
            self.calc_and_write_all_sims(train_tfidfs, test_tfidfs, self.simfile_path)
            print(str(datetime.datetime.now()),'written', self.simfile_path)
        else:
            print(str(datetime.datetime.now()),'read existing sim file')

        # calc preds
        if os.path.isfile( self.predsfile_path ) and not overwrite:
            print('WARNING: file already exists, overwriting: ' + self.predsfile_path)

        print(str(datetime.datetime.now()),'start prediction calc')
        self.calc_and_write_preds(self.simfile_path, train_tfidfs, test_tfidfs, self.num_neighbours, self.predsfile_path, self.write_num_recs)

        print(str(datetime.datetime.now()),'written', self.predsfile_path)
        print(str(datetime.datetime.now()),'############################ end knn_disk run(), took', round(time.time()-tstart), 'seconds')

    def get_res_name(self):
        return self.res_name

    def write_container_item_score(self, in_dict, filepath, header=''):
        with open(filepath, 'w') as f:
            w = csv.writer(f, delimiter=',')
            if header:
                w.writerow(header)
            for container in in_dict:
                for item in in_dict[container]:
                    score = in_dict[container][item]
                    w.writerow([container,item,score])

    def read_container_item_score(self, filepath, skip_header=False):
        with open(filepath, 'r') as f:
            r = csv.reader(f, delimiter=',')
            if skip_header:
                next(r)
            out_dict={}
            for row in r:
                container = int(row[0])
                item = int(row[1])
                score = float(row[2])
                if container not in out_dict:
                    out_dict[container] = {}
                out_dict[container][item] = score
        return out_dict

    def calc_and_write_all_sims(self, train_tfidfs, test_tfidfs, simfile_path):

        simfile = open(simfile_path, 'w')
        simwriter = csv.writer(simfile, delimiter=',')
        simwriter.writerow(['container_a','container_b','score'])

        test_containers_sorted=sorted(test_tfidfs.keys())
        
        progress=0
        tstart_calc=time.time()
        for container_a in test_containers_sorted:
            #print(container_a)
            #tstart = time.time()
            sims=self.calc_sims_one_container(container_a, test_tfidfs[container_a], train_tfidfs)
            #t_score= time.time() - tstart

            #tstart = time.time()
            for container_b in sorted(sims, key=sims.get, reverse=True):
                simwriter.writerow([container_a,container_b,sims[container_b]])
            #t_write= time.time() - tstart

            #print(container_a, container_b)
            #print(t_score, t_write)
            progress+=1
            if(progress%100==0):
                elapsed = round(time.time() - tstart_calc)
                remaining = round(elapsed / ( progress / len(test_containers_sorted) ) - elapsed)
                print(str(datetime.datetime.now()),'sim progress:',progress,'out of',len(test_containers_sorted),'. Elapsed:',elapsed,'Remaining:',remaining)

        simfile.close()


    def calc_and_write_preds(self, simfile_path, train_tfidfs, test_tfidfs, num_neighbours, predsfile_path, write_num_recs):
        predsfile = open(predsfile_path, 'w')
        predswriter = csv.writer(predsfile, delimiter=',')
        predswriter.writerow(['','confidence','playlist_id','track_id'])

        with open(simfile_path, 'r') as simfile:
            simreader = csv.reader(simfile, delimiter=',')
            # skip header
            next(simreader)
            last_test_container = -1
            count_num_neighbours = 0
            preds = {}
            count_for_preds_file = 0
            isfirst = True
            for row in simreader:
                #print('row:',row)
                test_container = int(row[0])
                train_container = int(row[1])
                sim = float(row[2])
                #print('extracted:',test_container,train_container,sim)

                if isfirst:
                    last_test_container = test_container
                    isfirst = False

                if test_container != last_test_container:
                    #print(preds)
                    count_num_recs_written = 0
                    for item in sorted(preds, key=preds.get, reverse=True):
                        if(count_num_recs_written>=write_num_recs):
                            break
                        predswriter.writerow([count_for_preds_file,preds[item],last_test_container,item])
                        #print('written preds:',[count_for_preds_file,preds[item],test_container,item])
                        count_num_recs_written+=1
                        count_for_preds_file+=1
                    #reset for next
                    last_test_container = test_container
                    count_num_neighbours = 0
                    preds = {}

                count_num_neighbours+=1
                if count_num_neighbours<=num_neighbours:
                    for item in train_tfidfs[train_container]:
                        # dont recommend tracks already in test playlist
                        if item in test_tfidfs[test_container]:
                            continue
                        score = train_tfidfs[train_container][item] * sim
                        if item in preds:
                            score += preds[item]
                        preds[item] = score

            #write last
            count_num_recs_written = 0
            for item in sorted(preds, key=preds.get, reverse=True):
                if(count_num_recs_written>=write_num_recs):
                    break
                predswriter.writerow([count_for_preds_file,preds[item],last_test_container,item])
                #print('written preds:',[count_for_preds_file,preds[item],test_container,item])
                count_num_recs_written+=1
                count_for_preds_file+=1

        predsfile.close()


    def data_as_dict(self, train, container_idx, item_idx):
        train_rows = train.values.tolist()
        train_dict = {}
        for row in train_rows:
            # 0 = artist, 1 = playlist, 2 = pos, 3= track_id
            # print(row[container_idx], row[item_idx])
            container = int(row[container_idx])
            item = int(row[item_idx])
            if container not in train_dict:
                train_dict[container] = set()
                
            train_dict[container].add(item)

        return train_dict

    def calc_train_test_tfidfs(self, train_container_items_list, test_container_items_list):
        
        train_inverse_map = {}
        for container in train_container_items_list.keys():
            for item in train_container_items_list[container]:
                if item not in train_inverse_map:
                    train_inverse_map[item] = set()
                train_inverse_map[item].add(container)

        train_num_containers = len(train_container_items_list.keys())
        train_item_idfs = self.calc_idfs(train_inverse_map, train_num_containers)
        
        test_item_idfs = {}
        for container in test_container_items_list.keys():
            for item in test_container_items_list[container]:
                if item not in test_item_idfs:
                    if item in train_item_idfs:
                        test_item_idfs[item] = train_item_idfs[item]
                    else:
                        print("TEST ITEM NOT IN TRAINING SET:",item)
                        test_item_idfs[item] = 1 / train_num_containers

        train_tfidfs=self.calc_tfidfs(train_container_items_list, train_item_idfs)
        test_tfidfs=self.calc_tfidfs(test_container_items_list, test_item_idfs)

        return train_tfidfs, test_tfidfs

    def calc_tfidfs(self, container_items_list, idfs):
        tfidfs = {}
        for container in container_items_list:
            tfidfs[container] = {}
            for item in container_items_list[container]:
                if self.tf_method=='one':
                    tf = 1
                elif self.tf_method=='ratio':
                    tf = 1 / len(container_items_list[container])
                elif self.tf_method=='ratio-s1':
                    tf = 1 / (len(container_items_list[container]) + 1)
                elif self.tf_method=='ratio-s10':
                    tf = 1 / (len(container_items_list[container]) + 10)
                elif self.tf_method=='ratio-s50':
                    tf = 1 / (len(container_items_list[container]) + 50)
                idf = idfs[item]
                if self.idf_method=='one':
                    tfidf = tf * 1
                elif self.idf_method=='log10':
                    tfidf = tf * log10(idf)
                elif self.idf_method=='plain':
                    tfidf = tf * idf
                tfidfs[container][item] = tfidf
        return tfidfs
        
    def calc_idfs(self, item_containers_list, num_containers):
        idfs = {}
        for item in item_containers_list:
            idfs[item] = num_containers / len(item_containers_list[item])
        return idfs


    # def count_container_size(self, in_dict):
    #     sizes = {}
    #     for container in in_dict:
    #         sizes[container] = len(in_dict[container])
    #     return sizes


    def calc_sims_one_container(self, container_a, item_weights_a, train_tfidfs):
        sims = {}
        for container_b in train_tfidfs:
            item_weights_b=train_tfidfs[container_b]
            if(container_a==container_b):
                continue

            score=self.sim(item_weights_a, item_weights_b)
            if(score!=0):
                sims[container_b]=score

        return sims


    def sim(self, item_weights_a, item_weights_b):
        similarity_method=self.similarity
        score = 0
        if similarity_method == 'jaccard':
            score = self.jaccard(item_weights_a.keys(),item_weights_b.keys(),self.sim_denom_add)
        elif similarity_method == 'cosine':
            score = self.cosine(item_weights_a,item_weights_b,self.sim_denom_add)
        else:
            print('ERROR: invalid similarity method')
            return None

        return score

    def jaccard(self, items_a, items_b, sim_denom_add):
        num = len(items_a & items_b)
        if(num == 0):
            return 0
        score = num / (len(items_a | items_b) + sim_denom_add)
        return score

    def cosine(self, item_weights_a, item_weights_b, sim_denom_add):
        num = 0
        for item in item_weights_a:
            if item in item_weights_b:
                num+=(item_weights_a[item] * item_weights_b[item])
        if(num == 0):
            return 0

        denom_a = sqrt(sum( w*w for w in item_weights_a.values()))
        denom_b = sqrt(sum( w*w for w in item_weights_b.values()))
        score = num / (denom_a + denom_b + sim_denom_add)
        return score
