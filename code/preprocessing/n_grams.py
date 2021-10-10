#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:52:42 2021

@author: rayaabuahmad
"""
import nltk
from nltk.util import bigrams, trigrams
from code.preprocessing.preprocessor import Preprocessor
import pandas as pd

class Ngrams(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output_col):
        # input column "tweet_no_punctuation_tokenized_no_stopwords_lemmatized"
        super().__init__([input_col], output_col)
        

    def _get_values(self, inputs):
        
        
        grams = []
        
        #loop over lemmatized tweets and add them into one list
        for lemmatized_tweet in inputs[0]:
            # extract ngrams for each tweet based on the given n
            apply_ngrams = list(trigrams(lemmatized_tweet))
            grams.append(apply_ngrams)
         
        #flatten all trigrams to be in one list rather than in a list of lists
        flat_grams = [gram for subgram in grams for gram in subgram]
        
        #count frequency of every trigram
        trigrams_series = pd.Series(flat_grams).value_counts()
            
        return trigrams_series
    