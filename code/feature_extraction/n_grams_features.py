#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 14:58:03 2021

@author: rayaabuahmad
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

class NgramsFeatures(FeatureExtractor):
    
   
    # constructor, default input_column would be 'tweet_no_hashtags_mentions_no_punctuation_emojis_tokenized_no_stopwords_lemmatized'
    def __init__(self, input_column, bigram):
        super().__init__([input_column], bigram)
   
    #get feature of the input bigram
    def _get_values(self, inputs): 
       
         results = []
         bigram = self.get_feature_name()
         for tweet in inputs[0]:
            
            result = False
            
            for index in range(len(tweet)):
                if (index != (len(tweet)-1)):
                    if tweet[index] == bigram[0] and tweet[index+1] == bigram[1]:
                        result = True
                    
            results.append(result)
        
                    
         #retun as feature
         results = np.array(results)
         results = results.reshape(-1,1)
         return results
