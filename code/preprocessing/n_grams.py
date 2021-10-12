#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 09:52:42 2021

@author: rayaabuahmad
"""
import nltk
from nltk.util import bigrams
from code.preprocessing.preprocessor import Preprocessor
import pandas as pd
import collections
import matplotlib.pyplot as plt
import numpy as np

class Ngrams(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output_col):
        # input column "tweet_no_punctuation_tokenized_no_stopwords_lemmatized"
        super().__init__([input_col], output_col)
        

    def _get_values(self, inputs):
        
        
        grams = []
        
        #loop over lemmatized tweets and add them into one list
        for lemmatized_tweet in inputs[0]:
            # extract bigrams for each tweet based on the given n
            apply_ngrams = list(bigrams(lemmatized_tweet))
            grams.append(apply_ngrams)
         
        #flatten all bigrams to be in one list rather than in a list of lists
        flat_grams = [gram for subgram in grams for gram in subgram]
        
        #count frequency of every bigram
        ngram_freq = collections.Counter(flat_grams)
        
        most_freq = ngram_freq.most_common(30)
        
        x_bigrams = ["("+",".join(bigram[0])+")" for bigram in most_freq]
        y_freq = [int(bigram[1]) for bigram in most_freq]
        
        
        plt.barh(x_bigrams, y_freq)
        plt.show()
        
        print(x_bigrams)
        print(y_freq)
    