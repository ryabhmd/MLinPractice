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
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_count".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the URL count based on the inputs
    def _get_values(self, inputs):
        
        mention_list = []
   
        for mention in inputs[0]:
            if mention == '[]':
                mention_list.append(0)
            else:
                mention_list.append(len(mention.split(",")))
            
        result = np.array(mention_list)
        result = result.reshape(-1,1)
        return result
    

    
# -*- coding: utf-8 -*-

