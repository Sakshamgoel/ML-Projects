import numpy as np

#from sklearn import cross_validation, preprocessing
from sklearn import preprocessing

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn import model_selection
from sklearn.metrics import classification_report, accuracy_score
from pandas.plotting import scatter_matrix

import matplotlib.pyplot as plt
import pandas as pd


# Loading the dataset

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
names = ['id', 'clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
		'signal_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

df = pd.read_csv(url, names = names)

# Preprocessing the data
# Basically, python will then ignore data points with '?'
df.replace('?', -999999, inplace = True)

# Since Id column is not really affecting our algorithm in any way
df.drop('id', 1, inplace = True)


#Dataset Visualizations
print(df.loc[698])


print(df.describe())

#df.hist(figsize = (100, 100))
#plt.show()

# Scatter plot matrix
#scatter_matrix(df, figsize = (18, 18))
#plt.show()

# Create X and Y datasets for training

X = np.array(df.drop('class', 1))
y = np.array(df['class'])

# SVM data needs to be scaled between 0...1 before implementation so this line does exactly that
X = np.array(preprocessing.StandardScaler().fit_transform(X))


X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

# Testing options
seed = 8
scoring = 'accuracy'

# Specify testing options
models = []

models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC(kernel = 'linear')))

# Evaluating each model in turn
results = []
names = []

for name, model in models:
	kfold = model_selection.KFold(n_splits = 10, shuffle = True, random_state = seed)
	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
	results.append(cv_results)
	names.append(name)

	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)


# Make predictions on validation dataset

for name, model in models:
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	print(name)
	print(accuracy_score(y_test, predictions))
	print(classification_report(y_test, predictions))


# Playing around
clf = SVC()

clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

example = np.array([[4,2,1,1,1,2,3,2,4]])
example = example.reshape(len(example), -1)
prediction = clf.predict(example)
print(prediction)