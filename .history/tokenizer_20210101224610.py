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

