#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessor that removes punctuation from the original tweet text.

Created on Wed Sep 29 09:45:56 2021

@author: lbechberger
"""

import string
from code.preprocessing.preprocessor import Preprocessor

# removes punctuation from the original tweet
# inspired by https://stackoverflow.com/a/45600350
class PunctuationRemover(Preprocessor):
    
    # constructor
    def __init__(self, input_col, output_col):
        # input column "tweet_no_hashtags_mentions", new output column
        super().__init__([input_col], output_col)
    
    # set internal variables based on input columns
    def _set_variables(self, inputs):
        # store punctuation for later reference
        self._punctuation = "[{}]".format(string.punctuation)
    
    # get preprocessed column based on data frame and internal variables
    def _get_values(self, inputs):
        # replace punctuation with empty string
        column = inputs[0].str.replace(self._punctuation, "")
        # added the punctuation mark ± and ’ which don't exist in the list
        column = column.str.replace("±", "")
        # remove whitespaces
        column = [" ".join(sentence.split()) for sentence in column]
        return column
    
    