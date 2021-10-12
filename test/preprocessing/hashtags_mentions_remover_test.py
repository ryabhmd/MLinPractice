#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 16:15:35 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.preprocessing.hashtags_mentions_remover import HashtagMentionRemover
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION_INPUT

class HashtagMentionRemoverTest(unittest.TestCase):
    
    def setUp(self):
       self.hashtags_mentions_remover = HashtagMentionRemover()
       
    
    def test_basic(self):
        input_text = "#Happy #LowKey @user111 whats up? #letshangout @user222 u 2"
        output_text = "whats up? u 2"
        
        input_df = pd.DataFrame()
        input_df[COLUMN_TWEET] = [input_text]
        
        result = self.hashtags_mentions_remover.fit_transform(input_df)
        self.assertEqual(result[COLUMN_PUNCTUATION_INPUT][0], output_text)
        
    def test_list(self):
        input_text = ["#Happy #LowKey @user111 whats up? #letshangout @user222 u 2", "@rrrr hiiii! #hi #whatisgoing on", "#machinelearning #datascience trying this out"]
        output_text = ["whats up? u 2" , "hiiii! on", "trying this out"]
          
        input_df = pd.DataFrame()
        input_df[COLUMN_TWEET] = input_text
         
        result = self.hashtags_mentions_remover.fit_transform(input_df)
        self.assertEqual(result[COLUMN_PUNCTUATION_INPUT][0], output_text[0])
       
if __name__ == '__main__':
   unittest.main()