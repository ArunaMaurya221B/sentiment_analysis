import pandas as p
import numpy as num
import matplotlib.pyplot as mat
import string
import collections

data = p.read_csv("/home/icts/sentiment analysis/train.csv").as_matrix()
#print data
temp = data = p.read_csv("/home/icts/sentiment analysis/train.csv").as_matrix()

tweets = temp[0:10000:,2]
#print tweets

counter = collections.Counter()

bag_of_words = []
for x in range(0, 10000):
	tweet = tweets[x]
	tweet = tweet.lstrip()
	#print tweet
	for c in string.punctuation:
		tweet = tweet.replace(c, "")#removing punctuations
		tweets[x] = tweet
	for d in string.digits:#removing digits
		tweet = tweet.replace(d, "")
		tweets[x] = tweet
	#print tweets[x]


for c in range(0, 10000):  #also counting spaces!
	tweet=tweets[c]
	bag_of_words.append(tweet.split(" "))
	counter.update(bag_of_words[c])
#print counter