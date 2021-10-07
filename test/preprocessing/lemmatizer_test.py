#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:53:03 2021

@author: rayaabuahmad
"""

import unittest
import pandas as pd
from code.preprocessing.lemmatizer import Lemmatizer

class LemmatizerTest(unittest.TestCase):
    
    def setUp(self):
       self.INPUT_COLUMN = "input"
       self.OUTPUT_COLUMN = "output"
       self.lemmatizer = Lemmatizer(self.INPUT_COLUMN, self.OUTPUT_COLUMN)
       
    #test if input col is the same
    def test_input_columns(self):
        self.assertListEqual(self.lemmatizer._input_columns, [self.INPUT_COLUMN])

     #test if output col is the same
    def test_output_column(self):
        self.assertEqual(self.lemmatizer._output_column, self.OUTPUT_COLUMN)
        
    #lemmatizer takes list of tokens as input, so all tests assume that tokenizer works and returns a correct list.
    def test_single_sentence(self):
        input_tokens = ['The', 'laughs', 'you', 'two', 'heard', 'were', 'triggered', 'by', 'memories', 'of', 'his', 'flying', 'moist', 'moisture', 'moisturize', 'moisturizing']
        output_tokens = ['the', 'laugh', 'you', 'two', 'heard', 'were', 'triggered', 'by', 'memory', 'of', 'his', 'flying', 'moist', 'moisture', 'moisturize', 'moisturizing']
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_tokens]
        
        lemmatized = self.lemmatizer.fit_transform(input_df)
        self.assertEqual(lemmatized[self.OUTPUT_COLUMN][0], output_tokens)
        
    def test_several_sentences_from_data(self):
        #took several actual tweets from data and lemmatized them myself to check output
        #after results, amended some things manually due to nltk lemmatizer constraints and errors
        input_tokens_list = [['10', 'Practical', 'BigData', 'Benefits', 'infographic', 'httpstcoMcYc02QcIQ', 'Analytics', 'DataScience', 'CX', 'innovation', 'DataDriven', 'RishabhSoft', 'httpstcoh2YeEhbSrP'], ['I', 'hope', 'twitter', 'puts', 'data', 'science', 'team', 'work', 'algorithms', 'Moments', 'editorial', 'selection', 'isnt', 'going', 'cut'], ['Neo4jBloom', 'Tips', 'Tricks', 'Domain', 'Knowledge', 'Experts', '–', 'GraphConnect', 'talk', 'Lju', 'Lazarevic', 'ElLazal', 'httpstcoCoIPexcmfK', 'DataVisualization', 'GraphDataModel']]
        output_tokens_1 = ['10', 'practical', 'bigdata', 'benefit', 'infographic', 'httpstcomcyc02qciq', 'analytics', 'datascience', 'cx', 'innovation', 'datadriven', 'rishabhsoft', 'httpstcoh2yeehbsrp']
        output_tokens_2 = ['i', 'hope', 'twitter', 'put', 'data', 'science', 'team', 'work', 'algorithm', 'moment', 'editorial', 'selection', 'isnt', 'going', 'cut']
        output_tokens_3 = ['neo4jbloom', 'tip', 'trick', 'domain', 'knowledge', 'expert', '–', 'graphconnect', 'talk', 'lju', 'lazarevic', 'ellazal', 'httpstcocoipexcmfk', 'datavisualization', 'graphdatamodel']
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = input_tokens_list
            
        lemmatized = self.lemmatizer.fit_transform(input_df)
        self.assertEqual(lemmatized[self.OUTPUT_COLUMN][0], output_tokens_1)
        self.assertEqual(lemmatized[self.OUTPUT_COLUMN][1], output_tokens_2)
        self.assertEqual(lemmatized[self.OUTPUT_COLUMN][2], output_tokens_3)
       
if __name__ == '__main__':
   unittest.main()
