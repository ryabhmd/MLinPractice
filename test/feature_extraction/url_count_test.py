#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: talkhiami
"""

import unittest
import pandas as pd
from code.feature_extraction.url_count import UrlCount

class UrlCountTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.url_count = UrlCount(self.INPUT_COLUMN)
               
        
    def test_input_columns(self):
        self.assertListEqual(self.url_count._input_columns, [self.INPUT_COLUMN])
   
    #testing
    def test_exist_url_count(self):
        input_text = "I really love this song! It's amazing! https://github.com/ryabhmd/MLinPractice"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.url_count.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    def test_nonexist_url_count(self):
        input_text = "I really love this song! It's amazing!"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.url_count.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()
