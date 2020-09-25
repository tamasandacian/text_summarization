import numpy as np
import networkx

class TextRank(object):
    def __init__(self, dt_matrix, td_matrix):
        self.dt_matrix = dt_matrix
        self.td_matrix = td_matrix
        self.similiarity_matrix = np.matmul(dt_matrix, td_matrix)

    def get_similarity_graph(self, similarity_matrix):
        """ Build similarity graph from similarity matrix """
        return networkx.from_numpy_array(similarity_matrix)
    
    def rank_sentences(self, similarity_graph):
        """ Compute pagerank scores for all sentences """
        scores = networkx.pagerank(similarity_graph)
        ranked_sents = sorted(((score, index) for index, score in scores.items()), reverse=True)
        return ranked_sents

    def get_top_sentence_indices(self, ranked_sents, n_sents):
        """ Get top sentence indices from ranked sentences & generate top sentences """
        top_sentence_indices = [ranked_sents[index][1] for index in range(n_sents)] 
        top_sentence_indices.sort()
        return top_sentence_indices