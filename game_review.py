import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas
from sklearn.model_selection import train_test_split

# Loading the data
games = pandas.read_csv("games.csv")

# Columns
#print(games.columns)

# Rows * Columns
#print(games.shape)

# Make a histogram of all the ratings in the average_rating columns
#plt.hist(games['average_rating'])
#plt.show()

# Print the first row of all the games with average_rating = 0 and that of games with average_rating > 0
#print(games[games['average_rating'] == 0].iloc[0])
#print(games[games['average_rating'] > 0].iloc[0])

# Removing games without user reviews
games = games[games['users_rated'] > 0]

# Removing any games with missing values
games.dropna(axis = 0)

#plt.hist(games['average_rating'])
#plt.show()

# Correlation matrix
corrmat = games.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = 0.8, square = True)
#plt.show()

#Get all columns from the dataFrame
columns = games.columns.tolist()

# Filter columns to remove data we do not want
columns = [c for c in columns if c not in ["bayes_average_rating", "average_rating", "type", "name", "id"]]

# Store the variable we'll be predicting on
target = "average_rating"


# Generate training and test datasets
train = games.sample(frac = 0.8, random_state = 1)

# Selecting everything that is not in train and putting it in test set
test = games.loc[~games.index.isin(train.index)]

# Dropping any naN or inifinite values, very important
train = train.dropna()
test = test.dropna()

# Rows and columns check for these sets
print(train.shape)
print(test.shape)

#importing linear regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Initialize model class
LR = LinearRegression()

# Fitting the training data in the model
x = LR.fit(train[columns], train[target])

# Generate predictions for test set
predictions = LR.predict(test[columns])

# LR Error
lr_error = mean_squared_error(predictions, test[target])

print(lr_error)

# Import the Random Forest Model
from sklearn.ensemble import RandomForestRegressor

# Initialize the model
RFR = RandomForestRegressor(n_estimators = 100, min_samples_leaf = 10, random_state = 1)

# Fit the data in it
RFR.fit(train[columns], train[target])

# Make Predictions
predictions = RFR.predict(test[columns])

# RFR error
rfr_error = mean_squared_error(predictions, test[target])
print(rfr_error)

# Some actual predictions
rating_LR = LR.predict(test[columns].iloc[0].values.reshape(1, -1))
rating_RFR = RFR.predict(test[columns].iloc[0].values.reshape(1, -1))

print(rating_LR)
print(rating_RFR)

# Actual value
print(test[target].iloc[0])