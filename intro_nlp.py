import pandas as pd
import numpy as np
import sklearn
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize

text = 'Hello Students, how are you doing today? The olympics are inspiring and Python is awesome. You look great today.'

#print(word_tokenize(text))

# Removing useless words from the data
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(text)

filtered_sentence = [w for w in word_tokens if not w in stop_words]

#print(filtered_sentence)

# Stemming words with NLTK
from nltk.stem import PorterStemmer

# ps = PorterStemmer()

# example_words = ['ride', 'rider', 'rides', 'riding']

# for w in example_words:
# 	print(ps.stem(w))

# # Stemming an entire sentence
# new_text = 'When riders ride horses, they often think of how cowboys rode horses.'

# new_tokens = word_tokenize(new_text)

# for w in new_tokens:
# 	print(ps.stem(w))

from nltk.corpus import udhr
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

# Training the PunktSentenceTokenizer
custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

# Tokenizing the sample text
tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Define a function that will tag each tokenized word with a part of the speech
def process_content():
	try:
		for i in tokenized[:50]:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)

			# combine the part of speech tag with a regular expression
			chunkGram = r"""Chunk: {<.*>+}
										}<VB.?|IN|DT|TO>+{"""
			chunkParser = nltk.RegexpParser(chunkGram)
			chunked = chunkParser.parse(tagged)

			# Print the nltk tree
			for subtree in chunked.subtrees(filter = lambda t: t.label() == 'Chunk'):
				print(subtree)

			# Draw the chunks with nltk
			#chunked.draw()

	except Exception as e:
		print(str(e))

def named_entity_recognition():
	try:
		for i in tokenized[:5]:
			words = nltk.word_tokenize(i)
			tagged = nltk.pos_tag(words)
			namedEnt = nltk.ne_chunk(tagged, binary = False)

			# Draw the chunks with nltk
			namedEnt.draw()

	except Exception as e:
		print(str(e))

#named_entity_recognition()


import random
from nltk.corpus import movie_reviews

# Build list of documents
documents = [(list(movie_reviews.words(fileid)), category)
			for category in movie_reviews.categories()
			for fileid in movie_reviews.fileids(category)]

# shuffle the documents
random.shuffle(documents)

# Getting to know the data

#print('Number of Documents: {}'.format(len(documents)))
#print('First Review: {}'.format(documents[0]))

all_words = []

for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

#print('Most Common Words: {}'.format(all_words.most_common(15)))
#print('The word happy: {}'.format(all_words['happy']))


# We will use the most common 4000 words
word_features = list(all_words.keys())[:4000]

# Build a find features function that will determine which of the 4000 word features are contained in a review
def find_features(document):
	words = set(document)
	features = {}

	for w in word_features:
		features[w] = (w in words)

	return features

# Let us use a negative review as an example
features = find_features(movie_reviews.words('neg/cv000_29416.txt'))

# for key, value in features.items():
# 	if value == True:
# 		print(key)

# Lets do it for all the documents
featuresets = [(find_features(rev), category) for (rev, category) in documents]

from sklearn import model_selection

# Define a seed for reproducibility
seed = 1

# split the data into training and testing dataset
training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)

print(len(training))
print(len(testing))

# How we use sklearn algorithms
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC

model = SklearnClassifier(SVC(kernel = 'linear'))

# Train the model on training data
model.train(training)

# Test on test dataset
accuracy = nltk.classify.accuracy(model, testing)

print('SVC accuracy: {}'.format(accuracy))
