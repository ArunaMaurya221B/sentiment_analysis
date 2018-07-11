""" sentiment analysis using Naiive Bayes classifier"""

import pandas as pd
import numpy as num
import matplotlib.pyplot as mat
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

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
	input = data, 
	encoding = 'utf-8', 
	decode_error = 'replace',
	strip_accents = None,
	stop_words = None,
	analyzer = 'word',
	lowercase = True,
	min_df = 2 
	)
#print vectorizer

x = vectorizer.fit_transform(data)
y = vectorizer.get_feature_names()

#print y
x = x.toarray()
y = num.array(y)

tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x)

#print x_train_tfidf

train_feature, test_feature, train_label, test_label = train_test_split(
	x,
	data_lables,
	train_size = 0.80,
	random_state=1234
	)

clf = MultinomialNB().fit(x_train_tfidf, data_lables)
pred = clf.predict(test_feature)
#print pred
#print test_label

print accuracy_score(test_label, pred)