import numpy as np
import sklearn
import pandas as pd
from matplotlib import pyplot as plt


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/molecular-biology/promoter-gene-sequences/promoters.data'

names = ['Class', 'id', 'Sequence']
data = pd.read_csv(url, names = names)

print(data.iloc[0])


classes = data.loc[:, 'Class']

# List of sequences
sequences = list(data.loc[:, 'Sequence'])

dataset = {}

for i, seq in enumerate(sequences):

	# Split into nucleotides and remove '/t/' in front
	nucleotides = list(seq)
	nucleotides = [x for x in nucleotides if x != '\t']

	# Append class assignment
	nucleotides.append(classes[i])

	# Add to dataset
	dataset[i] = nucleotides

print(dataset[0])

dframe = pd.DataFrame(dataset)

df = dframe.transpose()

#print(df.iloc[:5])

# Renaming the last column to 'Class'
df.rename(columns = {57: 'Class'}, inplace = True)

#print(df.iloc[:5])

# Knowing the data
df.describe()

# Record value counts for each sequence
series = []
for name in df.columns:
	series.append(df[name].value_counts())

info = pd.DataFrame(series)
details = info.transpose()
#print(details)

# Switch to numerical data using pd.get_dummies() function
numerical_df = pd.get_dummies(df)
numerical_df.iloc[:5]

# Remove one of the class columns and rename to simply 'Class'
df = numerical_df.drop(columns = ['Class_-'])

df.rename(columns = {'Class_+' : 'Class'}, inplace = True)
#print(df.iloc[:5])

# Import the algorithms
from sklearn.neighbors import KNeighborsClassifier # for a clustering based approach
from sklearn.neural_network import MLPClassifier # Multi-layer Perceptron for a neural network approach
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier # For decision tree approach
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn import model_selection

# Creating X and y datasets for training
X = np.array(df.drop(['Class'], 1))
y = np.array(df['Class'])

# Define a seed for reproducability
seed = 1

# Split the data in train and test dataset
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.25, random_state = seed)

# Define scoring technique
scoring = 'accuracy'

# Define models to train
names = ['K Nearest Neighbors', 'Gaussian Process', 'Decision Tree', 'Random Forest', 'Neural Net',
		'AdaBoost', 'Naive Bayes', 'SVM Linear', 'SVM RBF', 'SVM Sigmoid']


classifiers = [
			KNeighborsClassifier(n_neighbors = 3),
			GaussianProcessClassifier(1.0 * RBF(1.0)),
			DecisionTreeClassifier(max_depth = 5),
			RandomForestClassifier(max_depth = 5, n_estimators = 10, max_features = 1),
			MLPClassifier(alpha = 1),
			AdaBoostClassifier(),
			GaussianNB(),
			SVC(kernel = 'linear'),
			SVC(kernel = 'rbf'),
			SVC(kernel = 'sigmoid')
]

models = zip(names, classifiers)

# Evaluating each model in turn
results = []
names = []

# for name, model in models:
# 	kfold = model_selection.KFold(n_splits = 10, random_state = seed, shuffle = True)
# 	cv_results = model_selection.cross_val_score(model, X_train, y_train, cv = kfold, scoring = scoring)
# 	results.append(cv_results)
# 	names.append(name)

# 	msg = "{0}: {1} ({2})".format(name, cv_results.mean(), cv_results.std())
# 	print(msg)


for name, model in models:
	model.fit(X_train, y_train)
	predictions = model.predict(X_test)
	print(name)
	print(accuracy_score(y_test, predictions))
	print(classification_report(y_test, predictions))

