import sys
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
    def __init__(self, load_path, batch_size=16, minmax_doc_length=[300,2500], test_frac=.05, shuffle_scale=300, max_doc_loads=500, load_vec_flag=True):
        # Store parameters
        self.load_path = load_path
        self.batch_size = batch_size
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
        self.__schedule_pos = 0
        self.__schedule = None
        self.__initialize_training_schedule__()
        
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
        
        tot_queries = len(self.query_list)
        n_test = 0
        test_works   = []
        test_queries = []
        while float(len(test_queries)) / tot_queries < self.test_frac:
            # Randomly add a work to the test_works
            new_test_work = train_works.pop( np.random.randint(0,len(train_works)) )
            test_works = test_works + [new_test_work]
            
            # Add all the queries for that work
            test_queries += [q for q in self.query_list if q['doc'] == new_test_work]
        
        self.test_queries = test_queries
        
        # Re-create query_list so there is no overlap
        self.query_list = [q for q in self.query_list if q['doc'] in train_works]
        
        # Store the works going in to each
        self.__train_works = train_works
        self.__test_works = test_works
        
    def __initialize_training_schedule__(self):
        self.__schedule_pos = 0
        self.__weak_shuffle__()
    
    def __weak_shuffle__(self):
        list_len = len(self.query_list)
        def sortkey(X,obj):
            list_len = len(obj.query_list)
            X = X + np.random.normal(scale=obj.shuffle_scale)
            if X < 0:
                X = list_len - X
            if X > list_len:
                X = X - list_len
            return X
        
        # Give the data a random circular shift
        d = deque( range(list_len) )
        d.rotate(np.random.randint(0,list_len))
        shifted_sched = list(d)
        
        self.__schedule =\
            [x for x in sorted( shifted_sched, key=lambda X: sortkey(X,self) )]    
    
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
            
            return (vecs,tags,query['a'])
        
        # Package the next batch of examples
        schedule_head = self.__schedule_pos
        self.__current_vecs_tags_answers = []
        for i in range(self.batch_size):
            self.__current_vecs_tags_answers += [package_example(self,schedule_head)]
            schedule_head += 1
            if schedule_head >= len(self.query_list):
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
            
    # For permuting entities that ought not be memorized
    def permute_examples(self):

        def permute_example(self,vecs,tags,ans):
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
            return vecs.astype(theano.config.floatX), e_perm[ans-1]-1

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
        self.__schedule_pos = np.mod( self.__schedule_pos + self.batch_size, len(self.query_list) )
        
    def set_batch_size(self,new_batch_size):
        size_as_int = int(new_batch_size)
        if new_batch_size < 1:
            new_batch_size = 1
        self.batch_size = new_batch_size
        
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