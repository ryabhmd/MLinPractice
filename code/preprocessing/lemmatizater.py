#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that takes a list of tokens and returns them lemmatized with no duplicates.

Created on Thu Oct  7 13:02:19 2021

@author: rayaabuahmad
"""

from nltk.stem import WordNetLemmatizer
from code.preprocessing.preprocessor import Preprocessor

class Lemmatizer(Preprocessor):

    # constructor
    def __init__(self, input_col, output_col):
        # input column "tweet_no_punctuation_tokenized_no_stopwords", new output column
        super().__init__([input_col], output_col)
        
    def _get_values(self, inputs):
        
        #initiate wordnet lemmatizer
        wordnet_lemmatizer = WordNetLemmatizer()
        
        # initialize list to put filtered tweet in
        lemmatized_tweet_col = []
        
        #loop over tokenized tweets and lemmatize
        for tokenized_tweet in inputs[0]:
            lemmatized_tweet = [wordnet_lemmatizer.lemmatize(token.lower()) for token in tokenized_tweet]
            lemmatized_tweet_col.append(lemmatized_tweet)
        
        return lemmatized_tweet_col