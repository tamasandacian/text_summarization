from scipy.sparse.linalg import svds
import numpy as np

class LSA(object):
    def __init__(self, k, td_matrix):
        self.k = k
        self.td_matrix = td_matrix
        self.u, self.s, self.vt = svds(td_matrix, k)        # low_rank_svd
       
    def filter_singular_values(self, singular_values, sv_threshold):
        """ Filter singular values below threshold """
        min_sigma_value = max(singular_values) * sv_threshold
        singular_values[singular_values < min_sigma_value] = 0
        return singular_values
    
    def get_salience_scores(self, singular_values, topic_doc_matrix):
        """ Compute salience scores using singular values & topic document matrix """
        salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_doc_matrix)))
        return salience_scores

    def get_top_sent_indices(self, salience_scores, n_sents):
        """ Get top sentence indices from salience scores with first number of sentences """
        top_sent_indices = (-salience_scores).argsort()[:n_sents]
        top_sent_indices.sort()
        return top_sent_indices