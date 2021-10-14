#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of feature extractors.

Created on Wed Sep 29 11:00:24 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
import numpy as np
from code.feature_extraction.sentiment_analysis import SentimentAnalyzer
from code.feature_extraction.character_length import CharacterLength
from code.feature_extraction.feature_collector import FeatureCollector
from code.feature_extraction.url_count import UrlCount
from code.feature_extraction.mention_count import MentionCount
from code.feature_extraction.hashtag_count import HashtagCount
from code.feature_extraction.personal_story import PersonalStory
from code.feature_extraction.engage_keywords import EngageKeyword
from code.feature_extraction.ner_count import NERCount
from code.feature_extraction.n_grams_features import NgramsFeatures
from code.util import COLUMN_TWEET, COLUMN_LABEL, COLUMN_URL, COLUMN_MENTION, COLUMN_HASHTAG, COLUMN_STOPWORDS_INPUT


# setting up CLI
parser = argparse.ArgumentParser(description = "Feature Extraction")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output pickle file")
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
parser.add_argument("-i", "--import_file", help = "import an existing pipeline from the given location", default = None)
parser.add_argument("-c", "--char_length", action = "store_true", help = "compute the number of characters in the tweet")
parser.add_argument("-s", "--sentiment_analysis", action = "store_true", help = "compute the sentiment score of the tweet")
parser.add_argument("--sentiment_input", help = "input column to return sentiment analysis score from", default = COLUMN_TWEET)
parser.add_argument("-u", "--url_count", action = "store_true", help = "compute the number of URLs in the tweet")
parser.add_argument("-m", "--mention_count", action = "store_true", help = "compute the number of mentions in the tweet")
parser.add_argument("-ht", "--hashtag_count", action = "store_true", help = "compute the number of hashtags in the tweet")
parser.add_argument("-ps", "--personal_story", action="store_true", help = "check if the tweet contains one of the personal story keywords")
parser.add_argument("-ek", "--engage_keywords", action="store_true", help = "check if the tweet contains one of the engage keywords")
parser.add_argument("-n", "--ner_count", action = "store_true", help = "compute the number of named entities in the tweet")
parser.add_argument("--n_grams", action = "store_true", help = "for each of the 30 most frequent bigrams in the datset, check if it exists in each tweet")
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

if args.import_file is not None:
    # simply import an exisiting FeatureCollector
    with open(args.import_file, "rb") as f_in:
        feature_collector = pickle.load(f_in)

else:    # need to create FeatureCollector manually

    # collect all feature extractors
    features = []
    if args.char_length:
        # character length of original tweet (without any changes)
        features.append(CharacterLength(COLUMN_TWEET))
    if args.sentiment_analysis:
        # sentiment analysis nltk VADER compund score of original tweet (without any changes)
        features.append(SentimentAnalyzer(COLUMN_TWEET))
    if args.url_count:
        # URL array length of URL column
        features.append(UrlCount(COLUMN_URL))
    if args.mention_count:
        features.append(MentionCount(COLUMN_MENTION))
    if args.hashtag_count:
        features.append(HashtagCount(COLUMN_HASHTAG))
    if args.personal_story:
        features.append(PersonalStory(COLUMN_TWEET))
    if args.engage_keywords:
        features.append(EngageKeyword(COLUMN_TWEET))
    if args.ner_count:
        features.append(NERCount(COLUMN_STOPWORDS_INPUT))
        
    if args.n_grams:
        # unpickle the 30 most freq bigrams
        with open("data/preprocessing/bigrams.pickle", 'rb') as f_in:
            freq_bigrams = pickle.load(f_in)
        
        most_freq = freq_bigrams.most_common(30)
        
        for item in most_freq:
            features.append(NgramsFeatures("tweet_no_hashtags_mentions_no_punctuation_emojis_tokenized_no_stopwords_lemmatized", item[0]))
            

    # create overall FeatureCollector
    feature_collector = FeatureCollector(features)
    
    # fit it on the given data set (assumed to be training data)
    feature_collector.fit(df)


# apply the given FeatureCollector on the current data set
# maps the pandas DataFrame to an numpy array
feature_array = feature_collector.transform(df)

# get label array
label_array = np.array(df[COLUMN_LABEL])
label_array = label_array.reshape(-1, 1)

# store the results
results = {"features": feature_array, "labels": label_array, 
           "feature_names": feature_collector.get_feature_names()}
with open(args.output_file, 'wb') as f_out:
    pickle.dump(results, f_out)


# export the FeatureCollector as pickle file if desired by user
if args.export_file is not None:
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(feature_collector, f_out)