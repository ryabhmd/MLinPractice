#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:08:52 2021

@author: talkhiami
"""

import numpy as np
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the url count as a feature
class UrlCount(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_count".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # compute the URL count based on the inputs
    def _get_values(self, inputs):
        
        url_list = []
   
        for url in inputs[0]:
            if url == '[]':
                url_list.append(0)
            else:
                url_list.append(len(url.split(",")))
            
            
        result = np.array(url_list)
        result = result.reshape(-1,1)
        return result
