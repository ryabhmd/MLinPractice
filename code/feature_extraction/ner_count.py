#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Class that counts for each tweet how many NE there are and add this as a feature

Created on Thu Oct 14 11:07:32 2021

@author: rayaabuahmad
"""
import nltk
from nltk import pos_tag
import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

class NERCount(FeatureExtractor):
    
    # constructor, default input_column would be 'tweet_no_hashtags_mentions_no_punctuation_emojis_tokenized'
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_ne_count".format(input_column))
        
     # compute the Named Entities count based on the inputs
    def _get_values(self, inputs): 
         
        ne_count = []
        
        #iterate over each tweet in input column
        for tweet in inputs[0]:
            # extract pos and then chunck them 
            tag = pos_tag(tweet)
            ne_tree = nltk.ne_chunk(tag)
            
            #entities will have labels in nltk; so count all entities
            count = 0
            for entity in ne_tree:
                try:
                    entity.label()
                    count +=1
                except:
                    count +=0
        
        #retun as feature
        result = np.array(ne_count)
        result = result.reshape(-1,1)
        return result
             