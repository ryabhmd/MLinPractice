#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: talkhiami
"""

import unittest
import pandas as pd
from code.feature_extraction.hashtag_count import HashtagCount

class HashtagCountTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.hashtag_count = HashtagCount(self.INPUT_COLUMN)
               
        
    def test_input_columns(self):
        self.assertListEqual(self.hashtag_count._input_columns, [self.INPUT_COLUMN])
   
    #testing
    def test_exist_hashtag_count(self):
        input_text = "['analytics', 'ai', 'datascience', 'bigdata', 'dataethics', 'datascientists', 'orms', 'machinelearning']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.hashtag_count.fit_transform(input_df)
        self.assertEqual(result[0], 8)
        
    def test_nonexist_hashtag_count(self):
        input_text = "[]"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.hashtag_count.fit_transform(input_df)
        self.assertEqual(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()