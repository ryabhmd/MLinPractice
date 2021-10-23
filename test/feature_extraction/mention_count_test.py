#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: talkhiami
"""

import unittest
import pandas as pd
from code.feature_extraction.mention_count import MentionCount

class MentionCountTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.mention_count = MentionCount(self.INPUT_COLUMN)
               
        
    def test_input_columns(self):
        self.assertListEqual(self.mention_count._input_columns, [self.INPUT_COLUMN])
   
    #testing 
    def test_exist_mention_count(self):
        input_text = "I really love this song! It's amazing! @Adam @Sara"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.mention_count.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    def test_nonexist_mention_count(self):
        input_text = "I really love this song! It's amazing!"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.mention_count.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()
