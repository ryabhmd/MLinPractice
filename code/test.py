#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 23:36:39 2021

@author: talkhiami
"""

has_personal_story_list= []
LIST_PERSONAL_STORY_KEYWORDS = ['we','i']
print (LIST_PERSONAL_STORY_KEYWORDS)
tweet = 'I have to go'

has_personal_story = any(keyword in tweet for keyword in LIST_PERSONAL_STORY_KEYWORDS)
has_personal_story_list.append(has_personal_story)
print (tweet)
print (has_personal_story)