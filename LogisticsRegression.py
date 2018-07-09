import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.crossvalidation import train_test_split
from sklearn.linear_model import LogisticRegression

train_feature, test_feature, train__label, test_label = train_test_split(
	features_nd,
	data_labels,
	train_size = 0.80,
	random_state=1234
	)
data = []
data_labels = []
with open("./pos__tweets.txt") as f:
	for i in f:
		data.append(i)
		data_labels.append('pos')

with open("./neg_tweets.txt") as f:
	for i in f:
		data.append(i)
		data_labels.append('neg')


vectorizer = CountVectorizer(
	analyzer = 'word',
	lowercase = False
	)

features = vectorizer.fit_transform(
	data
)

features_nd = features.toarray()

log_model = LogisticRegression()
log_model = log_model.predict(x=train_feature, y=train__label)
y_pred = log_model.predict(test_feature)