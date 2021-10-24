#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD

=======
>>>>>>> ad5dca32c9a52a3a44a630f68b04e0f42af10dd2
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
<<<<<<< HEAD
        input_text = "I really love this song! It's amazing! #Music #Jazz"
=======
        input_text = "['analytics', 'ai', 'datascience', 'bigdata', 'dataethics', 'datascientists', 'orms', 'machinelearning']"
>>>>>>> ad5dca32c9a52a3a44a630f68b04e0f42af10dd2
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.hashtag_count.fit_transform(input_df)
<<<<<<< HEAD
        self.assertGreater(result[0], 0)
        
    def test_nonexist_hashtag_count(self):
        input_text = "I really love this song! It's amazing!"
=======
        self.assertEqual(result[0], 8)
        
    def test_nonexist_hashtag_count(self):
        input_text = "[]"
>>>>>>> ad5dca32c9a52a3a44a630f68b04e0f42af10dd2
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.hashtag_count.fit_transform(input_df)
<<<<<<< HEAD
        self.assertGreater(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()
=======
        self.assertEqual(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()
>>>>>>> ad5dca32c9a52a3a44a630f68b04e0f42af10dd2
