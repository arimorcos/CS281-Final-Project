import sys
import numpy as np
import cPickle
from collections import deque

class character_data_manager:
    """
    Your data butler.
    
    Built for the literary character text database.
    
    Parameters
    ----------
    load_path: string
        The path to the folder housing the formatted data (see 'queries to trainables.ipynb')
    [test_frac]: {0.05}
        Approximate fraction of queries to quarantine as test data
    [shuffle_scale]: {300}
        Strength of the shuffling by work. Larger = more random training schedule at the cost of more loading from disk
    [max_doc_loads]: {500}
        The number of documents that can have their vector sequences loaded at once. Increases memory footprint of butler.
    [load_vec_flag]: {True}
        Flag to control whether vectors are loaded (default) or computed when accessed. Please use the default.
    
    """
    def __init__(self,load_path,test_frac=.05,shuffle_scale=300,max_doc_loads=500,load_vec_flag=True):
        # Store parameters
        self.load_path = load_path
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
        self.__initialize_training_schedule__()
        
    
    # For doing a hard split of the data
    def __split_train_test__(self):
        # Randomly choose works and accumulate their queries until you have at least test_frac
        train_works = self.doc_dict.keys()
        
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
        
        
        
    def __initialize_training_schedule__(self):
        self.__schedule_pos = 0
        self.__weak_shuffle__()
    
    def __weak_shuffle__(self):
        list_len = len(self.query_list)
        def sortkey(x,obj):
            list_len = len(obj.query_list)
            x = x + np.random.normal(scale=obj.shuffle_scale)
            if x < 0:
                x = list_len - x
            if x > list_len:
                x = x - list_len
            return x
        
        # Give the data a random circular shift
        d = deque(self.query_list)
        d.rotate(np.random.randint(0,list_len))
        self.query_list = list(d)
        
        self.query_list =\
            [x for (y,x) in sorted( enumerate(self.query_list), key=lambda X: sortkey(X[0],self) )]    
    
    # We need a method to offer data. That's mostly what this is here for.
    def offer_data(self):
        """Used to pull a data/answer pair from the manager"""
        
        # Isolate the relevant query 
        query = self.query_list[self.__schedule_pos]
        
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
        
        # Store the vectors and tags so we can go right back to this point easily without re-loading
        self.__current_vecs = vecs
        self.__current_tags = tags
        self.__current_ans  = query['a']
        
        # Apply the permutation procedure, deal with particular words, and return the info!
        return self.permute_example()
    
    
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
    def permute_example(self):
        # Randomly permute good entities
        e_perm = np.random.permutation( np.arange(1,self.ent_vecs.shape[0]) )
        # And also bad entities
        b_perm = np.random.permutation( np.arange(0,self.bEnt_vecs.shape[0]) )
        
        # Create a copy of the vectors
        V = self.__current_vecs.copy()
        
        # Go through each token and make any final changes that are necessary
        for i,t in enumerate(self.__current_tags):
            if t:
                if t[0] == 0:
                    # An unknown word
                    V[i,:] = self.unknown_vec()
                if t[0] == 3:
                    # A query start indicator
                    V[i,:] = self.query_start_vec()
                if t[0] == 4:
                    # A blank (like, clue to the answer) indicator
                    V[i,:] = self.ent_vecs[0,:]
                if t[0] == 1:
                    # An entity. Give it it's randomly assigned vector
                    V[i,:] = self.ent_vecs[e_perm[t[1]-1],:]
                if t[0] == 2:
                    # A bad entity. Give it it's randomly assigned vector
                    V[i,:] = self.bEnt_vecs[b_perm[t[1]-1],:]
                    
        # Return the resulting vector-sequence+answer pair
        return V, e_perm[self.__current_ans-1]-1
    
    # For moving through the data
    def advance_schedule(self):
        self.__schedule_pos += 1
        if self.__schedule_pos > len(self.query_list):
            self.__initialize_training_schedule__()
            
            
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
    
    def get_current_query(self):
        return self.query_list[self.__schedule_pos]
    def get_current_doc(self):
        return self.doc_dict[self.query_list[self.__schedule_pos]['doc']]