#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:23:24 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.util import COLUMN_TWEET, COLUMN_PUNCTUATION

class PunctuationRemoverTest(unittest.TestCase):
    
    def setUp(self):
        self.punct_remover = PunctuationRemover()
        
    #test basic sentence
    def test_punct_remover_basic_sentence(self):
        input_text = "This!!! is... an?? example; sentence#$@&()%"
        output_text = "This is an example sentence"
            
        input_df = pd.DataFrame()
        input_df[COLUMN_TWEET] = [input_text]
            
        punct_removed = self.punct_remover.fit_transform(input_df)
        self.assertEqual(punct_removed[COLUMN_PUNCTUATION][0], output_text)
        
    #test more than one sentence
    def test_punct_remover_several_sentences(self):
        
        input_list = ["!!!---This ^^^ is *** the .,.,;][ first sentence", "!@#$%^&*()_+I am ....... sentence #2 :)", "~I found ± even 'more' punct = to check?"]
        input_df = pd.DataFrame()
        input_df[COLUMN_TWEET] = input_list
        
        output_text_1 = "This is the first sentence"
        output_text_2 = "I am sentence 2"
        output_text_3 = "I found even more punct to check"
        
        punct_removed = self.punct_remover.fit_transform(input_df)
        self.assertEqual(punct_removed[COLUMN_PUNCTUATION][0], output_text_1)
        self.assertEqual(punct_removed[COLUMN_PUNCTUATION][1], output_text_2)
        self.assertEqual(punct_removed[COLUMN_PUNCTUATION][2], output_text_3)
        
if __name__ == '__main__':
    unittest.main()