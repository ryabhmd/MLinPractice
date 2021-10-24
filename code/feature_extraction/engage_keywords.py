#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:08:52 2021

@author: talkhiami
"""

import numpy as np
from code.util import LIST_ENGAGEMENT_KEYWORDS
from code.feature_extraction.feature_extractor import FeatureExtractor
import ast

# class for extracting the hashtag count as a feature
class EngageKeyword(FeatureExtractor):
    
    # constructor, input column default is the lemmatized tweet
    def __init__(self, input_column):
        super().__init__([input_column], "engage_keywords")
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # check if the tweet contains one of the keywords from the imported list
    def _get_values(self, inputs):     
        
        has_engage_keywords_list = []
   
        #loop over each twwet in input and check if one of its lemmas exist in the keywords list
        for tweet in inputs[0]:
            has_engage_keywords = False
            tweet_list = ast.literal_eval(tweet)
            for lemma in tweet_list:
                if lemma in LIST_ENGAGEMENT_KEYWORDS:
                    has_engage_keywords = True
            has_engage_keywords_list.append(has_engage_keywords)
            
        result = np.array(has_engage_keywords_list)
        result = result.reshape(-1,1)
        return result
    
# -*- coding: utf-8 -*-

