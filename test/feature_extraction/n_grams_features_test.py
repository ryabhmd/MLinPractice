#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:14:13 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.feature_extraction.n_grams_features import NgramsFeatures

class NgramsFeaturesTest(unittest.TestCase):


    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.bigram = ('data', 'science')
        self.n_grams_feat = NgramsFeatures(self.INPUT_COLUMN, self.bigram)
        
    
    def test_input_columns(self):
        self.assertListEqual(self.n_grams_feat._input_columns, [self.INPUT_COLUMN])  


    def test_basic_ngram(self):
        
        input_list = [["I", "love", "data", "science", "!", "❤️"] , ["This", "is", "a", "sentence", "that", "does", "not", "include", "the", "bigram"]]
        output = [True, False]

        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = input_list
        
        result = self.n_grams_feat.fit_transform(input_df)    
        
        for index, item in enumerate(result):
            self.assertEqual(result[index], output[index])
        
if __name__ == '__main__':
   unittest.main()