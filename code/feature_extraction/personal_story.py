#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 23:08:52 2021

@author: talkhiami
"""

import numpy as np
from code.util import LIST_PERSONAL_STORY_KEYWORDS
from code.feature_extraction.feature_extractor import FeatureExtractor

# class for extracting the hashtag count as a feature
class PersonalStory(FeatureExtractor):
    
    # constructor
    def __init__(self, input_column):
        super().__init__([input_column], "{0}_personal_story".format(input_column))
    
    # don't need to fit, so don't overwrite _set_variables()
    
    # check if the tweet contains one at leat of the keywords
    def _get_values(self, inputs):     
        
        has_personal_story_list= []
        print (LIST_PERSONAL_STORY_KEYWORDS)
   
        for tweet in inputs[0]:
            l_tweet = tweet.lower()
            has_personal_story = any(keyword in l_tweet for keyword in LIST_PERSONAL_STORY_KEYWORDS)
            has_personal_story_list.append(has_personal_story)
            
        result = np.array(has_personal_story_list)
        result = result.reshape(-1,1)
        return result
    

    
# -*- coding: utf-8 -*-

