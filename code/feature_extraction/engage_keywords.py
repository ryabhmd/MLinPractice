#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:08:52 2021

@author: talkhiami
"""

import numpy as np
from code.util import LIST_ENGAGEMENT_KEYWORDS
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the hashtag count as a feature
class EngageKeyword(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_engage_keywords".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # check if the tweet contains one at leat of the keywords
    def _get_values(self, inputs):     
        
        contain_engage_keyword_list= []
   
        for tweet in inputs[0]:
            l_tweet = tweet.lower()
            contain_engage_keyword = any(keyword in l_tweet for keyword in LIST_ENGAGEMENT_KEYWORDS)
            contain_engage_keyword_list.append(contain_engage_keyword)
            
        result = np.array(contain_engage_keyword_list)
        result = result.reshape(-1,1)
        return result
    

    
# -*- coding: utf-8 -*-

