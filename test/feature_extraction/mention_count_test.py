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
    def test_several_mention_count(self):
        input_text = "[{'screen_name': 'norwichresearch', 'name': 'norwich research park', 'id': '171070404'}, {'screen_name': 'turinginst', 'name': 'the alan turing institute', 'id': '3697013177'}, {'screen_name': 'thequadram', 'name': 'quadram institute', 'id': '67343639'}, {'screen_name': 'earlhaminst', 'name': 'earlham institute', 'id': '93655345'}, {'screen_name': 'thesainsburylab', 'name': 'the sainsbury laboratory', 'id': '49119570'}]"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.mention_count.fit_transform(input_df)
        self.assertEqual(result[0], 5)
        
    #testing 
    def test_one_mention_count(self):
        input_text = "[{'screen_name': 'republic', 'name': 'republic', 'id': '811972460560019456'}]"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.mention_count.fit_transform(input_df)
        self.assertEqual(result[0], 1)
        
    def test_nonexist_mention_count(self):
        input_text = "[]"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.mention_count.fit_transform(input_df)
        self.assertEqual(result[0], 0)
        
        
    
if __name__ == '__main__':
   unittest.main()