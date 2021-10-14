#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for class NERCount

Created on Thu Oct 14 11:33:22 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.feature_extraction.ner_count import NERCount

class NERCountTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.ner_counter = NERCount(self.INPUT_COLUMN)
               
        
    def test_input_columns(self):
        self.assertListEqual(self.ner_counter._input_columns, [self.INPUT_COLUMN])
   
    def test_basic_sentence(self):
        
        text = "NASA awarded Elon Musk’s SpaceX a $2.9 billion contract to build the lunar lander."
        input_text = ["NASA", "awarded", "Elon", "Musk", "'","s", "SpaceX", "a", "$", "2 .9", "billion", "contract", "to", "build", "the", "lunar", "lander"]
        output_count = 2
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.ner_counter.fit_transform(input_df)
        
if __name__ == '__main__':
   unittest.main()