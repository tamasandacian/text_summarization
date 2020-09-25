import logging
from lsa import LSA
from text_rank import TextRank
from preprocessing import Preprocessing
from utility import Utility
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class Summarization(object):
    def __init__(self, lang_code, method="LSA", n_words=200, k=1, sv_threshold=0.5, min_df=0, max_df=.1, use_idf=True):
    
        self.lang_code = lang_code
        self.method = method
        self.n_words = n_words
        self.k = k                      # num topics
        self.sv_threshold = sv_threshold
        self.min_df = min_df
        self.max_df = max_df
        self.use_idf = use_idf
        self.valid_langs = ["en"]
 
        if self.lang_code in self.valid_langs:
            self.p = Preprocessing(lang_code=lang_code)
            self.tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, use_idf=use_idf)
                
    def generate_doc_term_matrix(self, norm_sents):
        """ Generate document term matrix from normalized sentences """
        dt_matrix = self.tfidf.fit_transform(norm_sents)
        dt_matrix = dt_matrix.toarray()
        return dt_matrix

    def generate_term_doc_matrix(self, dt_matrix):
        """ Generate term document matrix from document term matrix """
        td_matrix = dt_matrix.T 
        return td_matrix
    
    def generate_summary(self, sents, top_sentence_indices):
        """ Generate summary from original sentences using top sentence indices """
        sents = np.array(sents)
        summary = "\n".join(sents[top_sentence_indices])
        return summary

    def summarize(self, text, n_sents=3):
        """ Summarize a given text and get top sentences """
        try:
            prediction = dict()
            
            if text:
                if self.lang_code in self.valid_langs:
                    if Utility.get_doc_length(text) > self.n_words:
                        # generate sentences, normalized sentences from text
                        sents, norm_sents = self.p.text_preprocessing(text)
                        # generate doc-term-matrix, term-doc-matrix
                        dt_matrix = self.generate_doc_term_matrix(norm_sents) 
                        td_matrix = self.generate_term_doc_matrix(dt_matrix)
                        
                        if self.method == "LSA":
                            lsa = LSA(self.k, td_matrix)
                            term_topic_matrix, singular_values, topic_doc_matrix = lsa.u, lsa.s, lsa.vt
                            # remove singular values below given treshold
                            singular_values = lsa.filter_singular_values(singular_values, self.sv_threshold)
                            # get salience scores from top singular values & topic document matrix
                            salience_scores = lsa.get_salience_scores(singular_values, topic_doc_matrix)
                            # get the top sentence indices for summarization
                            top_sentence_indices = lsa.get_top_sent_indices(salience_scores, n_sents)
                            summary = self.generate_summary(sents, top_sentence_indices)
                        elif self.method == "TEXT_RANK":
                            tr = TextRank(dt_matrix, td_matrix)
                            # build similarity graph
                            similarity_matrix = tr.similiarity_matrix
                            similarity_graph = tr.get_similarity_graph(similarity_matrix)
                            # compute pagerank scores for all sentences
                            ranked_sents = tr.rank_sentences(similarity_graph)
                            # get the top sentence indices for summarization
                            top_sentence_indices = tr.get_top_sentence_indices(ranked_sents, n_sents)
                            summary = self.generate_summary(sents, top_sentence_indices)
                        else:
                            return "no method found"
                        
                        # apply cleaning for readability
                        summary = Utility.remove_multiple_whitespaces(summary)
                        summary = Utility.remove_trailing_whitespaces(summary)
                        prediction["summary"] = summary
                        prediction["message"] = "successful"
                    else:
                        return "required at least {} words".format(self.n_words)
                else:
                    return "language not supported".format()
            else:
                return "required textual content"
            return prediction
        except Exception:
            logging.error("exception occured", exc_info=True)