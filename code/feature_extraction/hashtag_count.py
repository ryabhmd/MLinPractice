#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:08:52 2021

@author: talkhiami
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the hashtag count as a feature
class HashtagCount(FeatureExtractor):
    
    # constructor, default col. is hashtags
    def __init__(self, input_column):
        super().__init__([input_column], "hashtags_count")
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the URL count based on the inputs
    def _get_values(self, inputs):
        
        hashtag_list = []
        
        #count number of hashtags in each tweet
        for hashtag in inputs[0]:
            if hashtag == '[]':
                hashtag_list.append(0)
            else:
                hashtag_list.append(len(hashtag.split(",")))
            
        result = np.array(hashtag_list)
        result = result.reshape(-1,1)
        return result
    

    
# -*- coding: utf-8 -*-

