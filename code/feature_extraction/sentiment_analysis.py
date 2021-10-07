#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract sentiment score of text from the given column.

Created on Thu Oct  7 10:40:58 2021

@author: rayaabuahmad
"""
# to run code one should run "nltk.download('vader_lexicon')" to download sentiment lexicon
from code.feature_extraction.feature_extractor import FeatureExtractor
from nltk.sentiment import SentimentIntensityAnalyzer
import numpy as np

# class for extracting the sentiment score of a text as a feature
class SentimentAnalyzer(FeatureExtractor):
    
    # constructor
    def __init__(self, input_col):
        # input column default "tweet", new output column
        super().__init__([input_col], "{0}_sentiment_score".format(input_col))
    
    # get sentiment score of the input col and return it as a new col
    def _get_values(self, inputs):

        #initiate nltk's VADER sentiment analyzer
        sia = SentimentIntensityAnalyzer()
        
        #initiate list that will include sentiment compound score of each tweet
        sentiment_scores = []
        
        #loop over tweets and add each one's score
        for tweet in inputs[0]:
            sentiment_scores.append(sia.polarity_scores(tweet)['compound'])
            
        result = np.array(sentiment_scores)
        result = result.reshape(-1,1)
            
        return result