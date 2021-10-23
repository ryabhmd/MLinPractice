#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:52:42 2021

@author: rayaabuahmad
"""
import pickle
from nltk.util import bigrams
from code.preprocessing.preprocessor import Preprocessor
import collections

class Ngrams(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output_col):
        # input column "tweet_no_punctuation_tokenized_no_stopwords_lemmatized"
        super().__init__([input_col], output_col)
        

    def _set_variables(self, inputs):
        
        
        grams = []
        
        #loop over lemmatized tweets and add them into one list
        for lemmatized_tweet in inputs[0]:
            # extract bigrams for each tweet based on the given n
            apply_ngrams = list(bigrams(lemmatized_tweet))
            grams.append(apply_ngrams)
         
        #flatten all bigrams to be in one list rather than in a list of lists
        flat_grams = [gram for subgram in grams for gram in subgram]
        
        updated_grams = []
        
        for gram in flat_grams:
            if not ((gram[0]=='data' and gram[1]=='science') or (gram[0]=='data' and gram[1]=='visualization') or (gram[0]=='data' and gram[1]=='analysis')):
                if ("http" not in gram[0]) and ("http" not in gram[1]):
                    updated_grams.append(gram)
        
        #count frequency of every bigram
        ngram_freq = collections.Counter(updated_grams)
        
        most_freq = ngram_freq.most_common(30)
        
        with open("data/preprocessing/bigrams.pickle", 'wb') as f_out:
             pickle.dump(ngram_freq, f_out)
    