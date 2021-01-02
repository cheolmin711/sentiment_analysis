# All modules that are required to import:
import numpy as np
import pandas as pd
import time

import requests
import bs4
import json
import re

import nltk
nltk.download('punkt')
nltk.download('twitter_samples')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

from nltk.corpus import twitter_samples

pos_tweet = twitter_samples.tokenized('positive_tweets.json')
neg_tweet = twitter_samples.tokenized('negative_tweets.json')

# filter out the stop words
# borrowed list of stop words from https://github.com/kavgan/stop-words/blob/master/terrier-stop.txt 

filepath = open("terrier-stop.txt", "r")
temp = filepath.read().split("\n")
stop_words = { key : 1 for key in temp }

