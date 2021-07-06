import sklearn
import numpy as np
import pandas as pd
import nltk

# Load the dataset of sms messages
df = pd.read_table('SMSSpamCollection', header = None, encoding = 'utf-8')

# Print useful information about the data set
#print(df.info())
#print(df.head())

# Check class distribution: checking how many unique values we have for each class
classes = df[0]
print(classes.value_counts())

# Preprocessing the data
from sklearn.preprocessing import LabelEncoder

# Convert 'ham', 'spam' to 0s and 1s
encoder = LabelEncoder()
y = encoder.fit_transform(classes)

# Storing the message data
text_messages = df[1]

# Use regex to replaces urls, emails, phone numbers, other numbers, symbols

# Replacing email addresses with 'emailaddr'
processed = text_messages.str.replace(r'^.+@[^\.].*\.[a-z]{2,}$', 'emailaddr', regex = True)

# Replace urls with 'webaddr'
processed = processed.str.replace(r'^http\://[a-zA-Z0-9\-\.]+\.[a-zA-Z]{2, 3}(/\S*)?$', 'webaddr', regex = True)

# Replace money symbols with 'moneysymb'
processed = processed.str.replace(r'£|\$', 'moneysymb', regex = True)

# Replace 10 digit phone numbers with 'phonenumber'
processed = processed.str.replace(r'^\(?[\d]{3}\)?[\s-]?[\d]{3}[\s-]?[\d]{3}$', 'phonenumber', regex = True)

# Replace normal numbers with 'number'
processed = processed.str.replace(r'\d+(\.\d+)?', 'number', regex = True)

# Remove punctuation
processed = processed.str.replace(r'[^\w\d\s]', ' ', regex = True)

# Replace white space between terms with single space
processed = processed.str.replace(r'\s+', ' ', regex = True)

# Removing leading and trailing white spaces
# processed = processed.strip() would perform the same job
processed = processed.str.replace(r'^\s+|\s+?$', '', regex = True)

# Change everything to lowercase
processed.str.lower()
#print(processed)

# Remove stop words from text messages
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

processed = processed.apply(lambda x: ' '.join(term for term in x.split() if term not in stop_words))

# Removing stems in words using porter stemmer
ps = nltk.PorterStemmer()

processed = processed.apply(lambda x: ' '.join(ps.stem(term) for term in x.split()))

#print(processed)

from nltk.tokenize import word_tokenize

# Creating a bag-of-words
all_words = []

for message in processed:
	words = word_tokenize(message)

	for w in words:
		all_words.append(w)

all_words = nltk.FreqDist(all_words)

print("Total number of words: {}".format(len(all_words)))
print("15 most common words: {}".format(all_words.most_common(15)))

# Use the 1500 most common words as features
word_features = list(all_words.keys())[:1500]

def find_features(message):
	words = word_tokenize(message)

	features = {}

	for word in word_features:
		features[word] = (word in words)


	return features

# Lets see an example

features = find_features(processed[0])
for key, value in features.items():
	if value == True:
		print(key)


# Find features for all messages
messages = zip(processed, y)

# Define a seed for reproducability
seed = 1
np.random.seed = seed
#np.random.shuffle(messages)

# Call find_features for all messages
featuresets = [(find_features(text), label) for (text, label) in messages]

# Split training and testing data using sklearn
from sklearn import model_selection

training, testing = model_selection.train_test_split(featuresets, test_size = 0.25, random_state = seed)

print('Training: {}'.format(len(training)))
print('Testing: {}'.format(len(testing)))

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Names of the models
names = ['K Nearest Neighbors', 'Decision Tree', 'Random Forest', 'Logistic Regression', 
		'SGD Classifier', 'Naive Bayes', 'SVM Linear']

# Initializing the models
classifiers = [
			KNeighborsClassifier(),
			DecisionTreeClassifier(),
			RandomForestClassifier(),
			LogisticRegression(),
			SGDClassifier(max_iter = 100),
			MultinomialNB(),
			SVC(kernel = 'linear')]


models = zip(names, classifiers)

#print(models)

# Wrap models in NLTK
from nltk.classify.scikitlearn import SklearnClassifier

# for name, model in models:
# 	nltk_model = SklearnClassifier(model)
# 	nltk_model.train(training)

# 	accuracy = nltk.classify.accuracy(nltk_model, testing) * 100
# 	print('Name: {}, Accuracy: {}'.format(name, accuracy))

d_models = {}

for name, model in models:
	d_models[name] = model


from sklearn.ensemble import VotingClassifier

nltk_ensemble = SklearnClassifier(VotingClassifier(estimators = d_models.items(), voting = 'hard', n_jobs = -1))
nltk_ensemble.train(training)

accuracy = nltk.classify.accuracy(nltk_ensemble, testing) * 100
print('Accuracy: {}'.format(accuracy))

# Make class label predictions
txt_features, labels = zip(*testing)

prediction = nltk_ensemble.classify_many(txt_features)

# Print a confusion matrix and classification report
print(classification_report(labels, prediction))

pd.DataFrame(confusion_matrix(labels, prediction), index = [['actual', 'actual'], ['ham', 'spam']], 
			columns = [['predicted', 'predicted'], ['ham', 'spam']])

