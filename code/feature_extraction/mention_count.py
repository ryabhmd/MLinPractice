#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:08:52 2021

@author: talkhiami
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the mentions count as a feature
class MentionCount(FeatureExtractor):
    
    # constructor, deafult col. is mentions
    def __init__(self, input_column):
        super().__init__([input_column], "mentions_count")
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the URL count based on the inputs
    def _get_values(self, inputs):
        
        mention_list = []
   
        # count number of mentions in each tweet
        for mention in inputs[0]:
            if mention == '[]':
                mention_list.append(0)
            else:
                mention_list.append(len(mention.split("{"))-1)
            
        result = np.array(mention_list)
        result = result.reshape(-1,1)
        return result
    

    
# -*- coding: utf-8 -*-

