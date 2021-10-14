#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 10:28:46 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.preprocessing.n_grams import Ngrams

class NgramsTest(unittest.TestCase):
    
    def setUp(self):
       self.INPUT_COLUMN = "input"
       self.OUTPUT_COLUMN = "output"
       self.n_grams = Ngrams(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
       
    def test_basic(self):
        input_list = [['the', 'work', 'partially', 'based', 'earlier', 'evaluation', 'community', 'higher', 'risk', 'could', 'benefit', 'enhanced', 'testing', 'relied', 'data', 'science', 'advisory', 'table', 'case', 'data', 'well', 'data', 'outbreak', '3'],
                      ['free', 'data', 'science', 'course', 'online', 'learn', 'new', 'useful', 'skill', 'via', 'republic', 'httpstcoohwqbvpd30'],
                      ['documentation', 'oftoverlooked', 'last', 'mile', 'shipping', 'data', 'science', 'project', 'elliotjg', 'take', 'look', 'best', 'practice', 'company', 'resource', 'create', 'stellar', 'readme', 'httpstco6jbf3vbx9c']
                      ]
        
        ouput_list = ['a','b']
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = input_list
        
        n_grams_result = self.n_grams.fit_transform(input_df)
        for gram in n_grams_result["output"]:
            print(gram)
        
if __name__ == '__main__':
    unittest.main()
        
       