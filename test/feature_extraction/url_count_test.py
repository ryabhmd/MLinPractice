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
        input_text = "['https://twitter.com/lennybronner/status/1381299032690163717']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.url_count.fit_transform(input_df)
        self.assertEqual(result[0], 1)
        
        input_text = "['https://twitter.com/lennybronner/status/1381299032690163717', 'https://studip.uni-osnabrueck.de/dispatch.php/my_courses', 'https://docs.python.org/3/library/unittest.html']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.url_count.fit_transform(input_df)
        self.assertEqual(result[0], 3)
        
    def test_nonexist_url_count(self):
        input_text = "[]"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.url_count.fit_transform(input_df)
        self.assertEqual(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()