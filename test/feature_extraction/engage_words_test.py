#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: talkhiami
"""

import unittest
import pandas as pd
from code.feature_extraction.engage_keywords import EngageKeyword

class PeersonalStoryTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.engage_keywords = EngageKeyword(self.INPUT_COLUMN)
               
        
    def test_input_columns(self):
        self.assertListEqual(self.engage_keywords._input_columns, [self.INPUT_COLUMN])
   
    #testing words count which are related to a engage_keywords
    def test_exist_engage_keywords_count(self):
        input_text = "Retweet my tweet please"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.engage_keywords.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    def test_nonexist_engage_keywords_count(self):
        input_text = "I really love this song! It's amazing!"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.engage_keywords.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()
