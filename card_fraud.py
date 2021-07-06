# Packages to be used
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import sklearn


# Load the data
data = pd.read_csv("creditcard.csv")

# For 'class', 1 = fraud, 0 = not fraud


# Getting useful information about the data
#print(data.columns)
#print(data.shape)
#print(data.describe())

# As the mean for the 'Class' column is 0.0017 (very close to zero), we have less fraud data,
# and would need to account for that.

# The data is too much
data = data.sample(frac = 0.1, random_state = 1)

print(data.shape)

# Plotting a histogram
#data.hist(figsize = (10, 10))
#plt.show()


# Getting the outlier_fraction
Fraud = data[data['Class'] == 1]
Valid = data[data['Class'] == 0]

outlier_fraction = len(Fraud) / float(len(Valid))

print('Fraud cases: {}'.format(len(Fraud)))
print('Valid cases: {}'.format(len(Valid)))
print('Outlier Fraction: {}'.format(outlier_fraction))

# Correlation matrix

corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))
sns.heatmap(corrmat, vmax = 0.8, square = True)
#plt.show()

columns = data.columns.tolist()

target = 'Class'

columns = [c for c in columns if c not in [target]]

X = data[columns]
y = data[target]

print(X.shape)
print(y.shape)

# Anomaly detection algorithms
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Define a random state
state = 1

# Define the outlier detection methods
classifiers = {
	"Isolation Forest": IsolationForest(max_samples = len(X),
										contamination = outlier_fraction, random_state = state),
	"Local Outlier Factor": LocalOutlierFactor(n_neighbors = 20, contamination = outlier_fraction)
}

# Fit the model
n_outliers = len(Fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):

	# Fit the data and tag outliers
	if clf_name == 'Local Outlier Factor':
		y_pred = clf.fit_predict(X)
		scores_pred = clf.negative_outlier_factor_
	else:
		clf.fit(X)
		scores_pred = clf.decision_function(X)
		y_pred = clf.predict(X)

	# Reshape the inlier and outlier values from -1, 1 to 1, 0 (1 for fraud, 0 for valid)
	y_pred[y_pred == 1] = 0
	y_pred[y_pred == -1] = 1

	n_errors = (y != y_pred).sum()

	print('{}: {}'.format(clf_name, n_errors))
	print(accuracy_score(y, y_pred))
	print(classification_report(y, y_pred))
