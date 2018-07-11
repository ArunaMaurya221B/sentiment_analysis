""" sentiment analysis using Naiive Bayes classifier"""

import pandas as pd
import numpy as num
import matplotlib.pyplot as mat
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

data = []
data_lables = []

with open("/home/icts/sentiment_analysis/pos_tweets.txt") as f:
	for i in f:
		data.append(i)
		data_lables.append('pos')

with open("/home/icts/sentiment_analysis/neg_tweets.txt") as f:
	for i in f:
		data.append(i)
		data_lables.append('neg')

files = ['/home/icts/sentiment_analysis/pos_tweets.txt',
		'/home/icts/sentiment_analysis/neg_tweets.txt' ]

#creating a count vectorizer
vectorizer = CountVectorizer(
	input = 'files', encoding = 'utf-8', decode_error = 'replace',
	strip_accents = 'None',
	stop_words = None,
	analyzer = 'word',
	lowercase = True,
	min_df = 2 
	)
#print vectorizer