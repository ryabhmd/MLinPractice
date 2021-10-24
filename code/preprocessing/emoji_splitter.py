#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that receive text and adds a space between alphanumeric characters and emojis.
#emoji module should be installed using 'pip install emoji'
Created on Fri Oct  8 13:02:30 2021

@author: rayaabuahmad
"""

import emoji
from code.preprocessing.preprocessor import Preprocessor


class EmojiSplitter(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output_col):
        # input column default "tweet_no_punctuation", new output column
        super().__init__([input_col], output_col)
        
    
    def _get_values(self, inputs):
        
        #initiate output column as empty list
        output_col = []
        
        #iterate over tweets in input
        for tweet in inputs[0]:
            #iterate over characters in tweet and check if it is part of keys in imported emoji dict
            for char in tweet:
                if char in emoji.UNICODE_EMOJI['en'].keys():
                    #if so, add spaces before and after it
                    tweet = tweet.replace(char,  " %s " %char)
            output_col.append(tweet)
                    
        # remove whitespaces from tweets
        output_col = [" ".join(sentence.split()) for sentence in output_col]
        
        return output_col
        
        