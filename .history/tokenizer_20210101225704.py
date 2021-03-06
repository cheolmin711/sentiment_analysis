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

# Goal of this part: Read through all positive / negative tweets, normalize and remove unnecessary words from tweets, then create actual dictionary-like to use for our dataset

# Convert all complex part-of-speech to basic words
# List of part-of-speech is in this link: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
# WordNetLemmatizer has a function lemmatize where you can convert complex part of speech words into basic forms
# Things to consider:
#   Remove all unnecessary words from normalized_neg_tweets / normalized_pos_tweets
#   1. Remove mentions(starts with @)
#   2. Remove links (starts with https:// or http:// )
#   3. Remove punctuation (starts with ! or ?)
#   4. Remove Stop-Words (words that do have little to no meaning and does not affect the context of the sentence) to make our dataset more concise
# Note that we are keeping emoji (i.e. :) or :( . That is because these emojis do actually show sentiment of the text context)
# If words are DETERMINERS (DT), COORDINATING CONJUCTIONS (CC), PREPOSITIONS (IN), PERSONAL / POSSESSIVE PRONOUNS (PRP / PRP$), or WH-PRONOUNS (WP) WH-ADVERB(WRB), we remove it (consider as Stop words)
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import string

normalizer = WordNetLemmatizer()
punctuation_and_stop_words = {'!': 1, '"': 1, '#': 1, '$': 1 ,'%': 1, '&': 1, "'": 1,'(': 1,')': 1,'*': 1,'+': 1,',': 1,'-': 1,'.': 1,':': 1,';': 1,'<': 1,'=': 1,'>': 1,'?': 1,'@': 1,'[': 1,']': 1,'^': 1,'_': 1,'`': 1,'{': 1,'|': 1,'}': 1,'~': 1,'https://': 1,'http://': 1}
stop_words_final = {**stop_words, **punctuation_and_stop_words}


def determiners(word):
    if word in stop_words_final:
        return False
    else:
        return True

def normalize(tweet_list):
    normalized_tweet = []
    for tweet in tweet_list:
        sentence = []
        for token, tag in pos_tag(tweet):
            # For Complex Noun words:
            if tag.startswith('NN'):
                new_tag = 'n'
            # For Complex Verb words
            elif tag.startswith('VB'):
                new_tag = 'v'
            # For stop-words
            elif tag.startswith('DT') or tag.startswith('CC') or tag.startswith('IN') or tag.startswith('PRP') or tag.startswith('PRP$') or tag.startswith('WP') or tag.startswith('WRB'):
                continue 
            # Every other words, convert them into adjective (pos = 'a')
            else:
                if determiners(token):
                    new_tag = 'a'
                else:
                    continue
            sentence.append(normalizer.lemmatize(token, new_tag))
        normalized_tweet.append(sentence)
    return normalized_tweet

normalized_pos_tweets = normalize(pos_tweet)
normalized_neg_tweets = normalize(neg_tweet)

# Now, store all positive / negative words into dictionary so it can be used as a guide for calculating sentiment for sentences

pos_words_dict = {}
neg_words_dict = {}

# Store all words into dictionary
for tweet in normalized_pos_tweets:
    for word in tweet:
        if word in pos_words_dict:
            temp = pos_words_dict[word.lower()]
            temp += 1
            pos_words_dict[word.lower()] = temp
        else:
            pos_words_dict[word.lower()] = 1

for tweet in normalized_neg_tweets:
    for word in tweet:
        if word in neg_words_dict:
            temp = neg_words_dict[word.lower()]
            temp += 1
            neg_words_dict[word.lower()] = temp
        else:
            neg_words_dict[word.lower()] = 1

# remove all emojis and leave only roman alphabets
pos_df = pd.DataFrame({'word': list(pos_words_dict.keys()), 'frequency': list(pos_words_dict.values())})
cleaned_pos_df = pos_df.loc[pos_df['word'].str.isalpha()]

neg_df = pd.DataFrame({'word': list(neg_words_dict.keys()), 'frequency': list(neg_words_dict.values())})
cleaned_neg_df = neg_df.loc[neg_df['word'].str.isalpha()]

# merge the two dataframes into one
merged = pd.merge(cleaned_pos_df, cleaned_neg_df, on='word', how='outer').fillna(0)
merged['frequency'] = merged['frequency_x'] - merged['frequency_y']

# scale the frequencies of each word between 1 to 5 for positive words, -1 to -5 for negative words
# if there is a duplicate word in both negative and positive dataset, take the difference in frequencies
# and consider it as a positive word if the positive frequency is higher, and vice versa

pos_max = merged['frequency'].max()
neg_min = abs(merged['frequency'].min())

def scaler(freq):

    if freq > 0:
        return freq * (4 / pos_max) + 1
    elif freq < 0:
        return freq * (4 / neg_min) - 1
merged = merged.assign(**{'scale':merged['frequency'].apply(scaler)})

# decided to drop words with a total frequency of zero, since they were words that appeared the same number of times as both negative and positive words
merged = merged.dropna()

# function to retrieve text from links and tokenize them into sentences.
def tokenize_sentence(url_dict):
    text = []
    title = []
    abstract = []
    for url in url_dict.:
        time.sleep(0.1)
        headers = {'user-agent': 'Mozilla/5.0 (X11; CrOS x86_64 8172.45.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/51.0.2704.64 Safari/537.36'}
        article = requests.get(url, headers = headers)
        soup = bs4.BeautifulSoup(article.content, 'html.parser')
        article_text_p = soup.find_all('p', attrs={'class': 'css-axufdj evys1bk0'})
        abstract_text_p = soup.find('p', attrs={'class': 'css-w6ymp8 e1wiw3jv0'})
        title_text_h1 = soup.find('h1', attrs={'data-test-id': 'headline'})
        temp = []
        title.append(title_text_h1.text)
        if abstract_text_p == None:
            abstract.append('Null Abstract')
        else:
            abstract.append(abstract_text_p.text)
    
        for item in article_text_p:
            temp.append(item.text)
        space = ' '
        article_text = space.join(temp)
        text.append(article_text)

    # Word Tokenization to sentences 

    tokenized_by_sentence = []
    for num in range(len(text)):
        del_quo = re.sub(",”", " ", text[num])
        del_quo_2  = re.sub("”", " ", del_quo)
        del_quo_3 = re.sub("“", "", del_quo_2)
        text_token = re.split("(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!|;|”)\s", del_quo_3)
        text_token.insert(0, abstract[num])
        text_token.insert(0, title[num])
        tokenized_by_sentence.append(text_token)
        
    return tokenized_by_sentence
