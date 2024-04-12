#index is already stemmed and stoplisted 
#query needs to be processed 
import numpy as np
import math
import operator
class Retrieve:
    
    # Create new Retrieve object storing index and term weighting
    # scheme
    def __init__(self, index, term_weighting):
        self.index = index
        self.term_weighting = term_weighting
        self.num_docs = self.compute_number_of_documents()
        self.lengths = self.compute_document_vector_lengths()
        
        
        
        

        
        
    # calling self.compute_number_of_documents returns number of documents (3204)
    def compute_number_of_documents(self):
        self.doc_ids = set() 
        for term in self.index:
            self.doc_ids.update(self.index[term])
        return len(self.doc_ids)




#this function takes doc frequency as a parameter and returns the idf
    def compute_idf(self, doc_freq):
        if doc_freq == 0:
            return 0
        else:
            return (math.log10((self.num_docs / doc_freq)))
    

    # this function returns a dictionary with the query terms as keys and freqency of term in query as values
    def create_query_dict(self, query):
        query_dict = {}
        for term in query:
            if term not in query_dict:
                query_dict[term] = 1
            else:
                query_dict[term] +=1
            
        return query_dict
    
    # this function applies the appropriate weighting to each term in the query dict
    def weight_query_dict(self, query_dict):
        query_dict_weight = {}

        for term in query_dict.keys():

            tf = query_dict[term]
            try:
                df = len(self.index[term])
                
                
                if self.term_weighting == 'tfidf':
                    idf = self.compute_idf(df)
                    query_dict_weight[term] = tf *idf
                    
                elif self.term_weighting == 'tf':
                    query_dict_weight[term] = tf
                else:
                    query_dict_weight[term] = 1
                    
                    
                
            except KeyError:
                if self.term_weighting == 'binary':
                    query_dict_weight[term] = 1
                else: 
                    query_dict_weight[term] = 0
        return query_dict_weight
            
    
    
    
    # this function computes the magnitude of vectors
    def vector_length(self, vector):
        
        return np.linalg.norm(vector)
        
    #  computes the cos q d between query vector and document vector
    def cos_q_d(self, query_vector, document_vector, document_vector_length):
        
        
        total = 0
        for i in range(len(query_vector)):
            total += query_vector[i] * document_vector[i]
        
        return total/(document_vector_length)
            

                 
              
    # this function returns a 2d array with dimensions num_docs * length of query dict
    # count vector[i] will return the frequency of terms in the ith document
    def create_vector(self, query_dict):
        count_vector = np.zeros((self.num_docs + 1, len(query_dict)))
        
        query = list(query_dict.keys())
        
        for i in range(len(query)):
            if query[i] in self.index.keys():
                term = query[i]
               
                term_count = self.index[term]
                df = len(self.index[term])
                
                for occurence in term_count.keys():
                    if self.term_weighting == 'tfidf':
                        idf = self.compute_idf(df)
                        count_vector[occurence][i] = term_count[occurence] *idf
                    elif self.term_weighting == 'tf':
                        count_vector[occurence][i] = term_count[occurence]
                    else:
                        count_vector[occurence][i] = 1
                        
                    
        return(count_vector)
                
            
    # this function returns an array of all of the terms in the document. For binary weighting, all the terms are given
    # a weight of 1
    # for tf and idf, the term frequency of each term is stored
    def compute_document_vector_lengths(self):
        document_vectors = {}
        lengths = {}
        
        
        for term in self.index:
            
            
            for doc in self.index[term]:

                if doc not in document_vectors:
                    if self.term_weighting == 'binary':
                        document_vectors[doc] = [1]
                        
                    else:
                        document_vectors[doc] = [self.index[term][doc]]
                        
                else:
                    if self.term_weighting == 'binary':
                        document_vectors[doc] += [1]
                        
                    else:
                        document_vectors[doc] += [self.index[term][doc]]
                        
        for doc in document_vectors.keys():
            lengths[doc] = self.vector_length(document_vectors[doc])
            
        return lengths
        


               
    
    
    
    def for_query(self, query):
        
        values = {}
        
        query_dict = self.create_query_dict(query)
        
        weighted = self.weight_query_dict(query_dict)
        
        # a list of all the weighted values
        q = list(weighted.values())
        
        document_vector = self.create_vector(query_dict)
        
        # this compares the given query to every document in the collection, calculating the cos q d between the query vector and document vector
        for d in range(1, len(document_vector)):
            
            document_vec_length = (self.lengths[d])
            
            
            
            values[d] = self.cos_q_d(q, (document_vector[d]), document_vec_length)
            
            
        
            
           
        #sorts the documents in descending order of cos q d values, returning a dictionary
        sorted_d = dict(sorted(values.items(), key=operator.itemgetter(1),reverse=True))
        
        #get list of documents sorted by cos q d values
        files =list(sorted_d.keys())
        
        # take the top 10 values from the sorted array
        top = files[0:10]
        
        return top

            
        
        
            
        
        
        
        
        
        
                    
                
                
            
        


