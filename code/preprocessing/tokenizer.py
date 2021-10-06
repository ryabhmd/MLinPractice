# -*- coding: utf-8 -*-

"""
Preprocessor that tokenizes the original tweet text.

@author: ryabhmd
"""

import nltk
from code.preprocessing.preprocessor import Preprocessor

#receives the tweet col and tokenizes it

class Tokenizer(Preprocessor):
    
    #constructor
    def __init__(self, input_col, output_col):
        """Initialize the Tokenizer with the given input and output column."""
        super().__init__([input_col], output_col)
    
    # don't need to implement _set_variables(), since no variables to set
    
    def _get_values(self, inputs):
        """Tokenize the tweet."""
        
        tokenized = []
        
        for tweet in inputs[0]:
            sentences = nltk.sent_tokenize(tweet)
            tokenized_tweet = []
            for sentence in sentences:
                words = nltk.word_tokenize(sentence)
                tokenized_tweet += words
            
            tokenized.append(str(tokenized_tweet))
        
        return tokenized
        
