# Deep learning Grid Search

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import sklearn
import keras
from sklearn.preprocessing import StandardScaler


# Importing the data
df = pd.read_csv('diabetes.csv')

#print(df.describe())

columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for col in columns:
	df[col].replace(0, np.NaN, inplace = True)

#print(df.describe())

df.dropna(inplace = True)

print(df.describe())

# Changing our data into input X and output y
dataset = df.values

# Last number is not inclusive so X has columns 0 to 7
X = dataset[:, 0:8]
y = dataset[:, 8].astype(int)

print(X.shape)
print(y.shape)

print(X[:, 5])

# Normalization
scaler = StandardScaler().fit(X)
X_standardized = scaler.transform(X)

data = pd.DataFrame(X_standardized)
print(data.describe())

#importing necessary packages

from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.layers import Dropout

# Defining the model
def create_model():
	model = Sequential()
	model.add(Dense(16, input_dim = 8, kernel_initializer = 'uniform', activation = 'linear'))
	model.add(Dense(2, input_dim = 8, kernel_initializer = 'uniform', activation = 'linear'))
	model.add(Dense(1, activation = 'sigmoid'))
	# Compile the model
	adam = Adam(learning_rate = 0.001)
	model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])

	return model


# Define a random seed
seed = 6
np.random.seed(seed)

# Create model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 20, verbose = 0)

# Define Grid search parameters
# neuron1 = [4, 8, 16]
# neuron2 = [2, 4, 8]


# # Build and fit the grid search
grid = GridSearchCV(estimator = model, cv = KFold(random_state = seed, shuffle = True), refit = True, verbose = 10)
grid_results = grid.fit(X_standardized, y)

# #summarize the results
# print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))

# means = grid_results.cv_results_['mean_test_score']
# stds = grid_results.cv_results_['std_test_score']
# params = grid_results.cv_results_['params']

# for mean, stdev, param in zip(means, stds, params):
# 	print('{0} ({1}) with: {2}'.format(mean, stdev, param))

y_pred = grid.predict(X_standardized)

# Generate a classification report
from sklearn.metrics import classification_report, accuracy_score

print(accuracy_score(y, y_pred))
print(classification_report(y, y_pred))

example = df.iloc[1]
print(example)

prediction = grid.predict(X_standardized[1].reshape(1, -1))
print(prediction)
