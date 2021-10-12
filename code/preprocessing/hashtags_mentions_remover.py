#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:23:55 2021

@author: rayaabuahmad
"""

from code.preprocessing.preprocessor import Preprocessor
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION_INPUT

class HashtagMentionRemover(Preprocessor):
    
    # constructor
    def __init__(self):
        # input column "tweet", new output column
        super().__init__([COLUMN_TWEET], COLUMN_PUNCTUATION_INPUT)
    
        
    # return a column that has the tweet without the text after hashtags and mentions
    def _get_values(self, inputs):
        
        column = []
        prefixes = ['@','#']
        for tweet in inputs[0]:
            words = []
            for word in tweet.split():
                word = word.strip()
                if word:
                    if word[0] not in prefixes:
                       words.append(word)
            column.append(' '.join(words))

        return column

   