#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility file for collecting frequently used constants and helper functions.

Created on Wed Sep 29 10:50:36 2021

@author: lbechberger
"""

# column names for the original data frame
COLUMN_TWEET = "tweet"
COLUMN_LIKES = "likes_count"
COLUMN_RETWEETS = "retweets_count"

# column names of novel columns for preprocessing
COLUMN_LABEL = "label"
COLUMN_PUNCTUATION = "tweet_no_punctuation"
COLUMN_EMOJIS_INPUT = "tweet_no_punctuation"
COLUMN_TOKENIZE_INPUT = "tweet_no_punctuation_emojis"
COLUMN_STOPWORDS_INPUT = "tweet_no_punctuation_emojis_tokenized"
COLUMN_LEMMATIZE_INPUT = "tweet_no_punctuation_emojis_tokenized_no_stopwords"

# suffixes to add to columns after preprocessing
SUFFIX_EMOJIS = "_emojis"
SUFFIX_TOKENIZED = "_tokenized"
SUFFIX_STOPWORDS = "_no_stopwords"
SUFFIX_LEMMATIZED = "_lemmatized"