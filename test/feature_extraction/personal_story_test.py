#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: talkhiami
"""

import unittest
import pandas as pd
from code.feature_extraction.personal_story import PersonalStory

class PeersonalStoryTest(unittest.TestCase):
    
    def setUp(self):
        self.INPUT_COLUMN = "input"
        self.personal_story = PersonalStory(self.INPUT_COLUMN)
               
        
    def test_input_columns(self):
        self.assertListEqual(self.personal_story._input_columns, [self.INPUT_COLUMN])
   
    #testing words count which are related to a personal story
    def test_personal_story_count(self):
        input_text = "I really love this song! It's amazing!"
        
        input_df = pd.DataFrame()
        input_df[self.INPUT_COLUMN] = [input_text]
        
        result = self.personal_story.fit_transform(input_df)
        self.assertGreater(result[0], 0)
        
    
if __name__ == '__main__':
   unittest.main()
