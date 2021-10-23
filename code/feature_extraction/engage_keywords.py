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
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_engage_keywords".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # check if the tweet contains one at leat of the keywords   
    def _get_values(self, inputs):     
        
        has_engage_keywords = False
   
        for tweet in inputs[0]:
            tweet_list = ast.literal_eval(tweet)
            for lemma in tweet_list:
                if lemma in LIST_ENGAGEMENT_KEYWORDS:
                    has_engage_keywords = True
            
        result = np.array(has_engage_keywords)
        result = result.reshape(-1,1)
        return result
    
# -*- coding: utf-8 -*-

