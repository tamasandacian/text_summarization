from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk import sent_tokenize
from utility import Utility
import numpy as np

class Preprocessing(object):
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.stopwords = []

        self.valid_langs = ["en"]
        if lang_code in self.valid_langs:
            if lang_code == "en":
                self.stopwords = stopwords.words("english")
        
    def remove_stopwords(self, text):
        """ Remove stopwords from text """
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token not in self.stopwords]
        text = " ".join(filtered_tokens)
        return text
    
    def normalize_text(self, text):
        """ Normalize by cleaning & removing stopwords from text """
        text = Utility.clean_text(text)
        text = self.remove_stopwords(text)
        return text

    def text_preprocessing(self, text):
        """ Generate list of sentences & normalized sentences from text """
        sents = sent_tokenize(text)
        normalize_corpus = np.vectorize(self.normalize_text)
        norm_sents = normalize_corpus(sents)
        return sents, norm_sents