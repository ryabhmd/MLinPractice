#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:23:55 2021

@author: rayaabuahmad
"""

import string
from code.preprocessing.preprocessor import Preprocessor


class HashtagMentionRemover(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output_col):
        # input column "tweet", new output column
        super().__init__([input_col], output_col)
    
    # function that takes text and return the text with no hashtags and mentions
    def strip_all_entities(text):
        entity_prefixes = ['@','#']
        for separator in  string.punctuation:
            if separator not in entity_prefixes :
                text = text.replace(separator,' ')
        words = []
        for word in text.split():
            word = word.strip()
            if word:
                if word[0] not in entity_prefixes:
                    words.append(word)
        return ' '.join(words)
        
    # return a column that has the tweet without the text after hashtags and mentions
    def _get_values(self, inputs):
        
        column = [self.strip_all_entities(tweet) for tweet in inputs[0]]
        return column

   