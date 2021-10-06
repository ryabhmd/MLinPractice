#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes stop words from a list of tokenized tweets.

Created on Wed Oct  6 17:59:19 2021

@author: rayaabuahmad
"""

from code.preprocessing.preprocessor import Preprocessor
from nltk.corpus import stopwords
import ast

class StopWordsRemover(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output_col):
        # input column "tweet", new output column
        super().__init__([input_col], output_col)
        
    # get preprocessed column based on data frame
    def _get_values(self, inputs):
        
        # get English stopwords from nltk
        stop_words = set(stopwords.words('english'))
        
        # initialize list to store all filtered tweets as final column
        col_stopwords_removed = []
        
        # initialize list to put filtered tweet in
        filtered_tweet = []
        
        # iterate over tokenized tweet and keep only content words
        for tokenized_tweet in inputs[0]:
            tokenized_tweet_list = ast.literal_eval(tokenized_tweet)
            filtered_tweet = [token for token in tokenized_tweet_list if token not in stop_words]
            col_stopwords_removed.append(filtered_tweet)
            
        return col_stopwords_removed
      
    