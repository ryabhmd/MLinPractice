#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runs the specified collection of preprocessing steps

Created on Tue Sep 28 16:43:18 2021

@author: lbechberger
"""

import argparse, csv, pickle
import pandas as pd
from sklearn.pipeline import make_pipeline
from code.preprocessing.punctuation_remover import PunctuationRemover
from code.preprocessing.emoji_splitter import EmojiSplitter
from code.preprocessing.tokenizer import Tokenizer
from code.preprocessing.stop_words_remover import StopWordsRemover
from code.preprocessing.lemmatizer import Lemmatizer
from code.preprocessing.n_grams import Ngrams
from code.util import COLUMN_EMOJIS_INPUT, COLUMN_TOKENIZE_INPUT, SUFFIX_EMOJIS, SUFFIX_TOKENIZED, COLUMN_STOPWORDS_INPUT, SUFFIX_STOPWORDS, COLUMN_LEMMATIZE_INPUT, SUFFIX_LEMMATIZED


# setting up CLI
parser = argparse.ArgumentParser(description = "Various preprocessing steps")
parser.add_argument("input_file", help = "path to the input csv file")
parser.add_argument("output_file", help = "path to the output csv file")
parser.add_argument("-p", "--punctuation", action = "store_true", help = "remove punctuation")
parser.add_argument("-m", "--emoji_splitter", action = "store_true", help = "add spaces before and after each emoji in a given column of text")
parser.add_argument("--emoji_input", help = "input column to split emojis from", default = COLUMN_EMOJIS_INPUT)
parser.add_argument("-t", "--tokenize", action = "store_true", help = "tokenize given column into individual words")
parser.add_argument("--tokenize_input", help = "input column to tokenize", default = COLUMN_TOKENIZE_INPUT)
parser.add_argument("-s", "--stopwords", action = "store_true", help = "remove stopwords from a given column")
parser.add_argument("--stopwords_input", help = "input column to remove stopwords from", default = COLUMN_STOPWORDS_INPUT)
parser.add_argument("-l", "--lemmatize", action = "store_true", help = "lemmatize token from a given list of tokens")
parser.add_argument("--lemmatize_input", help = "input column to lemmatize from", default = COLUMN_LEMMATIZE_INPUT)
parser.add_argument("-e", "--export_file", help = "create a pipeline and export to the given location", default = None)
args = parser.parse_args()

# load data
df = pd.read_csv(args.input_file, quoting = csv.QUOTE_NONNUMERIC, lineterminator = "\n")

# collect all preprocessors
preprocessors = []

if args.punctuation:
    preprocessors.append(PunctuationRemover())

if args.emoji_splitter:
    preprocessors.append(EmojiSplitter(args.emoji_input, args.emoji_input + SUFFIX_EMOJIS))
    
if args.tokenize:
    preprocessors.append(Tokenizer(args.tokenize_input, args.tokenize_input + SUFFIX_TOKENIZED))
    
if args.stopwords:
    preprocessors.append(StopWordsRemover(args.stopwords_input, args.stopwords_input + SUFFIX_STOPWORDS))
    
if args.lemmatize:
    preprocessors.append(Lemmatizer(args.lemmatize_input, args.lemmatize_input + SUFFIX_LEMMATIZED))

# call all preprocessing steps
for preprocessor in preprocessors:
    df = preprocessor.fit_transform(df)
    
triigrams_series = Ngrams("tweet_no_punctuation_emojis_tokenized_no_stopwords_lemmatized", "output").fit_transform(df)
trigram_100 = triigrams_series[:20]

# store the results
df.to_csv(args.output_file, index = False, quoting = csv.QUOTE_NONNUMERIC, line_terminator = "\n")

trigram_100.plot.barh(color='blue', width=.9, figsize=(12, 8))

# create a pipeline if necessary and store it as pickle file
if args.export_file is not None:
    pipeline = make_pipeline(*preprocessors)
    with open(args.export_file, 'wb') as f_out:
        pickle.dump(pipeline, f_out)