#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 17:31:41 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.preprocessing.emoji_splitter import EmojiSplitter


class EmojiSplitterTest(unittest.TestCase):
    
    def setUp(self):
       self.INPUT_COLUMN = "input"
       self.OUTPUT_COLUMN = "output"
       self.emoji_splitter = EmojiSplitter(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
       
    #test if input col is the same
    def test_input_columns(self):
        self.assertListEqual(self.emoji_splitter._input_columns, [self.INPUT_COLUMN])

    #test if output col is the same
    def test_output_column(self):
        self.assertEqual(self.emoji_splitter._output_column, self.OUTPUT_COLUMN)
        
    #test that checks basic and complex emojis in one sentence
    def test_basic_sentence(self):
        input_text = "I👧👩🏽 love💓 💞my cat🐱."
        output_text = "I 👧 👩 🏽 love 💓 💞 my cat 🐱 ."
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        emojis_split = self.emoji_splitter.fit_transform(input_df)
        self.assertEqual(emojis_split[self.OUTPUT_COLUMN][0], output_text)
        
            
if __name__ == '__main__':
    unittest.main()
        
           
    