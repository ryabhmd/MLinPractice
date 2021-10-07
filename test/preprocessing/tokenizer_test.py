# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:23:24 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.preprocessing.tokenizer import Tokenizer

class TokenizerTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.OUTPUT_COLUMN = "output"
        self.tokenizer = Tokenizer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
    
    #test if input col is the same
    def test_input_columns(self):
        self.assertListEqual(self.tokenizer._input_columns, [self.INPUT_COLUMN])

     #test if output col is the same
    def test_output_column(self):
        self.assertEqual(self.tokenizer._output_column, self.OUTPUT_COLUMN)

    #test normal sentence
    def test_tokenization_single_sentence(self):
        input_text = "This is an example sentence"
        output_text = "['This', 'is', 'an', 'example', 'sentence']"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
        
    #test sentence with clitic (Let's) and some foreign words
    def test_tokenization_sentence_with_clitic_and_foreign_words(self):
        input_text = "@myke: Let's test these words: resumé España München français"
        output_text = "['@', 'myke', ':', 'Let', \"'s\", 'test', 'these', 'words', ':', 'resumé', 'España', 'München', 'français']"
            
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
            
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
        
    #test sentence with asterisks and some punctuation (even though real application removes punt first)
    def test_tokenization_sentence_with_asterisk_and_punct(self):
        input_text = "This is a, *weird sentence with *asterisks in it."
        output_text = "['This', 'is', 'a', ',', '*', 'weird', 'sentence', 'with', '*', 'asterisks', 'in', 'it', '.']"
                
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
                
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
        
    #test sentence with emojis
    def test_tokenization_sentence_with_emoji(self):
        input_text = "This is a sentence with so many emojis ❤️🐼😍🐱 🐶 ."
        output_text = "['This', 'is', 'a', 'sentence', 'with', 'so', 'many', 'emojis', '❤️🐼😍🐱', '🐶', '.']"
                    
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
                    
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text)
        
    #test more than one sentence
    def test_more_sentences(self):
        input_list = ["This is the first sentence in a list of sentences!", "I am sentence #2 😈", "This'll be the last sentence of all :)"]
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = input_list
        output_text_1 = "['This', 'is', 'the', 'first', 'sentence', 'in', 'a', 'list', 'of', 'sentences', '!']"
        output_text_2 = "['I', 'am', 'sentence', '#', '2', '😈']"
        output_text_3 = "['This', \"'ll\", 'be', 'the', 'last', 'sentence', 'of', 'all', ':', ')']"
        
        tokenized = self.tokenizer.fit_transform(input_df)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][0], output_text_1)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][1], output_text_2)
        self.assertEqual(tokenized[self.OUTPUT_COLUMN][2], output_text_3)

if __name__ == '__main__':
    unittest.main()