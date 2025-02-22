import sys
import os
import numpy as np
import cPickle
from collections import deque
import theano

class character_data_manager:
    """
    Your data butler.
    
    Built for the literary character text database.
    
    Parameters
    ----------
    load_path: string
        The path to the folder housing the formatted data (see 'queries to trainables.ipynb')
    [batch_size]: {16}
        Number of examples to offer at once
    [minmax_doc_length]: {[300, 2500]}
        Minimum and maximum lengths, respectively, of documents to include in training/test
    [test_frac]: {0.05}
        Approximate fraction of queries to quarantine as test data
    [shuffle_scale]: {300}
        Strength of the shuffling by work. Larger = more random training schedule at the cost of more loading from disk
    [max_doc_loads]: {500}
        The number of documents that can have their vector sequences loaded at once. Increases memory footprint of butler.
    [load_vec_flag]: {True}
        Flag to control whether vectors are loaded (default) or computed when accessed. Please use the default.
    
    """
    def __init__(self, load_path, batch_size=16, stride=16, perms_per=1, minmax_doc_length=[300,2500], test_frac=.05, shuffle_scale=300, max_doc_loads=500, load_vec_flag=True):
        if os.path.isfile(load_path):
            print 'load_path points to a file. Trying to use it to revive a saved butler...'
            how_make_butler = 'load'
        elif os.path.isdir(load_path):
            # This is a directory, presumably where the raw info lives. Make a new data butler.
            how_make_butler = 'new'
        else:
            raise BaseException('load_path was not a file nor directory. What the fuck, dude?')

        if how_make_butler == 'new':
            # Store parameters
            self.load_path = load_path
            self.batch_size = batch_size
            self.stride = stride
            self.perms_per = perms_per
            self.minmax_doc_length = minmax_doc_length
            self.test_frac = test_frac
            self.shuffle_scale = shuffle_scale
            self.max_doc_loads = max_doc_loads
            self.load_vec_flag = load_vec_flag

            # Load in the document dictionary, query_list, and the entity lookups
            with open(load_path + 'document_dictionary.pkl','rb') as f:
                self.doc_dict = cPickle.load(f)
            with open(load_path + 'query_list.pkl','rb') as f:
                self.query_list = cPickle.load(f)
            with open(load_path + 'entity_vectors.pkl','rb') as f:
                self.ent_vecs = cPickle.load(f)
            with open(load_path + 'bad_entity_vectors.pkl','rb') as f:
                self.bEnt_vecs = cPickle.load(f)

            # First things first, split the data
            self.__split_train_test__()

            # Initialize a training schedule
            self.__loaded_docs = []
            self.__schedule_pos_train = 0
            self.__schedule_train = None
            self.__initialize_training_schedule__()

            self.__schedule_pos_test = 0
            self.__schedule_test = range(len(self.test_queries))

            self.__schedule = None
            self.__schedule_pos = None

            # Default to pulling from the training data
            self.__curr_source = None
            self.pull_from_train()

            self.loop_test_flag = False

        elif how_make_butler == 'load':
            # Retreive the save dictionary from the fallen butler
            with open(load_path, 'rb') as f:
                save_dict = cPickle.load(f)

            # Re-instate parameters:
            self.load_path = save_dict['load_path']
            self.batch_size = save_dict['batch_size']
            self.stride = save_dict['stride']
            self.perms_per = save_dict['perms_per']
            self.minmax_doc_length = save_dict['minmax_doc_length']
            self.test_frac = save_dict['test_frac']
            self.shuffle_scale = save_dict['shuffle_scale']
            self.max_doc_loads = save_dict['max_doc_loads']
            self.load_vec_flag = save_dict['load_vec_flag']
            # Re-instate train/test details
            self.test_indices = save_dict['test_indices']
            self.__schedule_test = save_dict['__schedule_test']
            self.__schedule_pos_test = save_dict['__schedule_pos_test']
            self.__test_works = save_dict['__test_works']
            self.train_indices = save_dict['train_indices']
            self.__schedule_train = save_dict['__schedule_train']
            self.__schedule_pos_train = save_dict['__schedule_pos_train']
            self.__train_works = save_dict['__train_works']
            self.__curr_source = save_dict['__curr_source']
            self.loop_test_flag = save_dict['loop_test_flag']
            # Vectors
            self.ent_vecs = save_dict['ent_vecs']
            self.bEnt_vecs = save_dict['bEnt_vecs']

            # Get back the big data
            with open(self.load_path + 'document_dictionary.pkl', 'rb') as f:
                self.doc_dict = cPickle.load(f)
            with open(self.load_path + 'query_list.pkl', 'rb') as f:
                self.query_list = cPickle.load(f)

            # Get back the training and test queries
            self.test_queries = [self.query_list[i] for i in self.test_indices]
            self.train_queries = [self.query_list[i] for i in self.train_indices]

            # Lastly, set everything to whatever we were last working with
            if self.__curr_source == 'train':
                self.pull_from_train()
            elif self.__curr_source == 'test':
                self.pull_from_test()

    def pull_from_train(self):
        """
        Set the manager to pull from the training set
        """
        self.query_list = self.train_queries
        self.__schedule = self.__schedule_train
        self.__schedule_pos = self.__schedule_pos_train
        self.__curr_source = 'train'
        print 'Now offering: Training data!'

    def pull_from_test(self):
        """
        Set the manager to pull from the test set
        """
        self.query_list = self.test_queries
        self.__schedule = self.__schedule_test
        self.__schedule_pos = self.__schedule_pos_test
        self.__curr_source = 'test'
        print 'Now offering: Test data!'

    def save_butler(self, save_path):
        save_dict = {
            'test_indices': self.test_indices,
            '__schedule_test': self.__schedule_test,
            '__schedule_pos_test': self.__schedule_pos_test,
            '__test_works': self.__test_works,
            'train_indices': self.train_indices,
            '__schedule_train': self.__schedule_train,
            '__schedule_pos_train': self.__schedule_pos_train,
            '__train_works': self.__train_works,
            '__curr_source': self.__curr_source,
            'loop_test_flag': self.loop_test_flag,
            'load_path': self.load_path,
            'batch_size': self.batch_size,
            'stride': self.stride,
            'perms_per': self.perms_per,
            'ent_vecs': self.ent_vecs,
            'bEnt_vecs': self.bEnt_vecs,
            'minmax_doc_length': self.minmax_doc_length,
            'test_frac': self.test_frac,
            'shuffle_scale': self.shuffle_scale,
            'max_doc_loads': self.max_doc_loads,
            'load_vec_flag': self.load_vec_flag,
        }
        with open(save_path, 'wb') as f:
            cPickle.dump(save_dict, f)

    # For doing a hard split of the data
    def __split_train_test__(self):
        """
        Randomly choose works and accumulate their queries until you have at least test_frac
        """
        # These are all the works we have data for
        train_works = self.doc_dict.keys()
        
        # Exclude documents that are just too short or too long
        train_works = [w for w in train_works
                       if  len(self.doc_dict[w]['tags']) >= self.minmax_doc_length[0]
                       and len(self.doc_dict[w]['tags']) <= self.minmax_doc_length[1]]



        tot_queries = len([i for i, q in enumerate(self.query_list) if q['doc'] in train_works])
        test_works = []
        test_queries_and_indices = []
        while float(len(test_queries_and_indices)) / tot_queries < self.test_frac:
            # Randomly add a work to the test_works
            new_test_work = train_works.pop( np.random.randint(0,len(train_works)) )
            test_works = test_works + [new_test_work]
            
            # Pull out train/test queries and their indices in the original
            test_queries_and_indices = [(q, i) for i, q in enumerate(self.query_list) if q['doc'] in test_works]

        # Same thing for train queries
        train_queries_and_indices = [(q, i) for i, q in enumerate(self.query_list) if q['doc'] in train_works]

        self.test_queries, self.test_indices = zip(*test_queries_and_indices)
        self.train_queries, self.train_indices = zip(*train_queries_and_indices)
        
        # Store the works going in to each
        self.__train_works = train_works
        self.__test_works = test_works
        
    def __initialize_training_schedule__(self):
        self.__schedule_pos = 0
        self.__weak_shuffle__()

    def reset_test_schedule(self):
        self.__schedule_pos_test = 0
    
    def __weak_shuffle__(self):
        list_len = len(self.train_queries)
        def sortkey(X, obj):
            list_len = len(obj.train_queries)
            X = X + np.random.normal(scale=obj.shuffle_scale)
            if X < 0:
                X = list_len - X
            if X > list_len:
                X = X - list_len
            return X
        
        # Give the data a random circular shift
        d = deque(range(list_len))
        d.rotate(np.random.randint(0, list_len))
        shifted_sched = list(d)
        
        self.__schedule_train =\
            [x for x in sorted(shifted_sched, key=lambda X: sortkey(X, self))]
    
    # We need a method to offer data. That's mostly what this is here for.
    def offer_data(self):
        """Used to pull a data/answer pair from the manager"""
        def package_example(self,schedule_head):
            # Isolate the relevant query 
            query = self.query_list[self.__schedule[schedule_head]]

            # The data will be in the form of a vector sequence
            # So, we need to load and/or retreive this example's vector and tag sequences
            # Retreive the document vectors
            d_vec = self.__get_doc_vec__(query['doc'])
            # Retreive the query vectors
            q_vec = self.__get_query_vec__(query)
            # Combine them
            vecs = np.concatenate( (d_vec,q_vec) )
            # Retreive the tags
            tags = self.doc_dict[query['doc']]['tags'] + query['tags']
            
            return (vecs, tags, query['a'])
        
        # Package the next batch of examples
        schedule_head = self.__schedule_pos
        self.__current_vecs_tags_answers = []
        for i in range(self.batch_size):
            if self.__curr_source == 'test' and not self.loop_test_flag and schedule_head >= len(self.query_list):
                print 'Reached end of test data. Enable test looping or reset the test schedule to get more data.'
                break
            else:
                for p in range(self.perms_per):
                    self.__current_vecs_tags_answers += [package_example(self,schedule_head)]
                schedule_head += 1
                if schedule_head >= len(self.query_list):
                    if self.__curr_source == 'train' or self.loop_test_flag:
                        schedule_head = 0
        
        # Apply the permutation procedure, deal with particular words, and return the info!
        return self.permute_examples()
    
    # For managing the rather large data
    def __get_doc_vec__(self,doc_name):
        if type(self.doc_dict[doc_name]['vecs']) != type(None):
            return self.doc_dict[doc_name]['vecs']
        
        else:
            # Load the document vectors in
            self.doc_dict[doc_name]['vecs'] = self.__get_vec__(self.doc_dict[doc_name])

            self.__loaded_docs += [doc_name]

            if len(self.__loaded_docs) > self.max_doc_loads:
                # Reset the "oldest" load
                self.doc_dict[self.__loaded_docs[0]]['vecs'] = None
                self.__loaded_docs = self.__loaded_docs[1:]
            
            return self.doc_dict[doc_name]['vecs']
    
    def __get_query_vec__(self,query):
        return self.__get_vec__(query)
        
    # The method for accessing vectors when we don't want to keep them in memory
    def __get_vec__(self,dic):
        if self.load_vec_flag:
            # Load pre-computed vectors
            with open(self.load_path + dic['loc'],'rb') as f:
                return cPickle.load(f)
            
        else:
            # Use spacy to compute vectors
            # NOTE: THIS IS MUCH SLOWER !!!
            return np.array([ t.vector for t in nlp(dic['text']) ])

    # Convert entity vectors to one-hot
    def convert_ent_to_one_hot(self):
        """
        Sets each row of ent_vecs and bEnt_vecs (excluding index 0) to a one-hot vector with the one-value at the
        index of the row
        """
        # Modify good entitiy vectors
        num_ent, vec_size = self.ent_vecs.shape
        for ent_ind in range(1, num_ent):
            temp_vec = 0.005*np.ones(vec_size)
            temp_vec[ent_ind - 1] = 1.9999
            #temp_vec *= (1/np.linalg.norm(temp_vec))
            self.ent_vecs[ent_ind, :] = temp_vec

        # Modify bad entitiy vectors
        num_ent, vec_size = self.bEnt_vecs.shape
        for ent_ind in range(1, num_ent):
            temp_vec = 0.005*np.ones(vec_size)
            temp_vec[ent_ind - 1 + 180] = 1.9999
            #temp_vec *= (1 / np.linalg.norm(temp_vec))
            self.bEnt_vecs[ent_ind, :] = temp_vec

    # For permuting entities that ought not be memorized
    def permute_examples(self):

        def permute_example(self, vecs, tags, ans):
            # Randomly permute good entities
            e_perm = np.random.permutation( np.arange(1,self.ent_vecs.shape[0]) )
            # And also bad entities
            b_perm = np.random.permutation( np.arange(0,self.bEnt_vecs.shape[0]) )

            # Go through each token and make any final changes that are necessary
            for i,t in enumerate(tags):
                if t:
                    if t[0] == 0:
                        # An unknown word
                        vecs[i,:] = self.unknown_vec()
                    if t[0] == 3:
                        # A query start indicator
                        vecs[i,:] = self.query_start_vec()
                    if t[0] == 4:
                        # A blank (like, clue to the answer) indicator
                        vecs[i,:] = self.ent_vecs[0,:]
                    if t[0] == 1:
                        # An entity. Give it it's randomly assigned vector
                        vecs[i,:] = self.ent_vecs[e_perm[t[1]-1],:]
                    if t[0] == 2:
                        # A bad entity. Give it it's randomly assigned vector
                        vecs[i,:] = self.bEnt_vecs[b_perm[t[1]-1],:]

            # Return the resulting vector-sequence+answer pair
            return 17.32*vecs.astype(theano.config.floatX), e_perm[ans-1]-1

        def ans_to_onehot(corr_ans, num_options):
            n = len(corr_ans)
            onehot = np.zeros( (num_options, n ) )
            onehot[ corr_ans, np.arange(n) ] = 1.
            return onehot.astype(theano.config.floatX)

        # Permute each example in the list of current ones
        perms = [permute_example(self,v,t,a)
                 for v,t,a in self.__current_vecs_tags_answers]

        # Pull out the vectors list and answers list
        vecs, answers = zip(*perms)

        # We need 3 things: a matrix with all the vector sequences, a list of sequence lengths, and onehot answer matrix
        # Make the onehot answer matrix
        onehot_answers = ans_to_onehot( answers, self.ent_vecs.shape[0]-1 )

        # Make the list of sequence lengths
        seq_lengths = [v.shape[0] for v in vecs]

        # The slightly tricker thing is to zero-pad and stack the vectors into a matrix
        max_length = max(seq_lengths)
        # (actually, not that tricky)
        pad_seqs = [np.pad(V,((0,max_length-L),(0,0)),mode='constant') for V,L in zip(vecs,seq_lengths)]

        return np.dstack(pad_seqs), seq_lengths, onehot_answers
        
    # For moving through the training
    def advance_schedule(self):
        if self.__curr_source == 'test' and not self.loop_test_flag:
            # Don't loop.
            proposed_pos = self.__schedule_pos + self.stride
            if proposed_pos >= len(self.test_queries):
                self.__schedule_pos = len(self.test_queries)
                print 'You are already seeing the end of the test data. Either reset the test schedule or set ' \
                      'loop_test_flag to True.'
            else:
                self.__schedule_pos = proposed_pos
        else:
            self.__schedule_pos = np.mod( self.__schedule_pos + self.stride, len(self.query_list) )
        
    def set_batch_size(self, new_batch_size, disp=True):
        size_as_int = int(new_batch_size)
        if size_as_int < 1:
            size_as_int = 1
        self.batch_size = size_as_int
        if self.batch_size < self.stride:
            self.stride = self.batch_size
        if disp:
            print 'Batch Size = {};  Stride = {}'.format(self.batch_size, self.stride)
        
    def set_stride(self, new_stride, disp=True):
        stride_as_int = int(new_stride)
        if stride_as_int < 1:
            stride_as_int = 1
        if stride_as_int > self.batch_size:
            stride_as_int = self.batch_size
            print 'Cannot set stride to be greater than batch size!'
        if stride_as_int < 1:
            stride_as_int = 1
        self.stride = stride_as_int
        if disp:
            print 'Batch Size = {};  Stride = {}'.format(self.batch_size, self.stride)
        
    def set_perms_per(self, new_perms_per, disp=True):
        pp_as_int = int(new_perms_per)
        if pp_as_int < 1:
            pp_as_int = 1
        self.perms_per = pp_as_int
        if disp:
            print '{} examples per offer: Batch Size = {}  *  Permutations per = {}'.format(
                self.batch_size*self.perms_per,self.batch_size, self.perms_per)
        
    # Vectors for things we have to make up
    def unknown_vec(self):
        return np.zeros(300).astype('float32')
    
    def query_start_vec(self):
        O = np.ones(300).astype('float32')
        return O / np.sqrt(300.)
    
    # For getting to know your butler
    def num_loaded(self):
        return len(self.__loaded_docs)
    
    def loaded_docs(self):
        return self.__loaded_docs[:]
    
    def vec_memory_footprint(self):
        D = 0
        for k,v in self.doc_dict.iteritems():
            try:
                nB = v['vecs'].nbytes
            except:
                nB = sys.getsizeof(v['vecs'])
            D += nB
        return D / 1e6
    
    def get_batch_size(self):
        return self.batch_size
    
    def get_stride(self):
        return self.stride
    
    def get_perms_per(self):
        return self.perms_per
    
    def get_current_schedule(self):
        curr_schedule = []
        schedule_head = self.__schedule_pos
        for i in range(self.batch_size):
            curr_schedule += [self.__schedule[schedule_head]]
            schedule_head += 1
            if schedule_head >= len(self.query_list):
                schedule_head = 0
        return curr_schedule
    
    def get_current_queries(self):
        return [self.query_list[i] for i in self.get_current_schedule()]
    
    def get_current_doc(self):
        curr_q = self.get_current_queries()
        return [self.doc_dict[q['doc']] for q in curr_q]
    
    def get_train_works(self):
        return self.__train_works
    
    def get_test_works(self):
        return self.__test_works