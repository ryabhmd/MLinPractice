#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 21:26:39 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.feature_extraction.sentiment_analysis import SentimentAnalyzer

class SentimentAnalyzerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.sentiment_analyzer = SentimentAnalyzer(self.INPUT_COLUMN)
               
        
    def test_input_columns(self):
        self.assertListEqual(self.sentiment_analyzer._input_columns, [self.INPUT_COLUMN])
   
    #testing a generic positive sentence to see that the sentiment score is bigger than 0.
    def test_positive_emotion(self):
        input_text = "I really love this song! It's amazing!"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.sentiment_analyzer.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    #testing a generic negative sentence to see that the sentiment score is smaller than 0.
    def test_negative_emotion(self):
        input_text = "This is really awful I hate it :("
            
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
            
        result = self.sentiment_analyzer.fit_transform(input_df)
        self.assertLess(result[0], 0)
        
    #testing a generic neutral sentence (fact) to see that the sentiment score is 0.    
    def test_neutral_emotion(self):
        input_text = "The sun rises in the east."
            
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
            
        result = self.sentiment_analyzer.fit_transform(input_df)
        self.assertEqual(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()
