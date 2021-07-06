import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn

from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

fig, axs = plt.subplots(3, 3, figsize = (12, 12))
plt.gray()

# Loop through each subplot and add mnist images
for i, ax in enumerate(axs.flat):
	ax.matshow(X_train[i])
	ax.axis('off')
	ax.set_title('Number: {}'.format(y_train[i]))

# Display the figure
#fig.show()
#plt.show()

# Image preprocessing

# Converting the dataset into a 1D array
X = X_train.reshape(len(X_train), -1)
y = y_train

# Normalize the data to 0-1
X = X.astype(float) / 255.0

from sklearn.cluster import MiniBatchKMeans

n_digits = len(np.unique(y_test))
print(n_digits)

# Initialize KMeans model
kmeans = MiniBatchKMeans(n_clusters = n_digits)

# Fit the model to training data
kmeans.fit(X)

# This doesn't mean that they are the integers they represent, they just belong to that cluster that they have
# have been assigned
print(kmeans.labels_[:20])

# Associates most probable label with each cluster in KMeans model
# returns dictionary of clusters assigned to each label
def infer_cluster_labels(kmeans, actual_labels):

	inferred_labels = {}

	for i in range(kmeans.n_clusters):

		# Find index of points in cluster
		labels = []
		index = np.where(kmeans.labels_ == i)

		# Append actual labels for each point in cluster
		labels.append(actual_labels[index])

		# Determine most common label
		if len(labels[0]) == 1:
			counts = np.bincount(labels[0])
		else:
			counts = np.bincount(np.squeeze(labels))

		# Assign the cluster to a value in the inferred_labels dictionary
		if np.argmax(counts) in inferred_labels:
			# Append the new number to the existing array at this key
			inferred_labels[np.argmax(counts)].append(i)
		else:
			# Create a new array for this key
			inferred_labels[np.argmax(counts)] = [i]

	return inferred_labels


# Determines labels for each array, depending on the cluster it has been assigned to
# Returns predicted label for each array
def infer_data_labels(X_labels, cluster_labels):

	# Empty array of len(X)
	predicted_labels = np.zeros(len(X_labels)).astype(np.uint8)

	for i, cluster in enumerate(X_labels):
		for key, value in cluster_labels.items():
			if cluster in value:
				predicted_labels[i] = key

	return predicted_labels

cluster_labels = infer_cluster_labels(kmeans, y)
X_clusters = kmeans.predict(X)
predicted_labels = infer_data_labels(X_clusters, cluster_labels)
print(predicted_labels[:20])
print(y[0:20])

# Optimizing and evaluating the clustering algorithm

from sklearn import metrics

def calculate_metrics(estimator, data, labels):

	# Calculate and print metrics
	print('Number of clusters: {}'.format(estimator.n_clusters))
	print('Intertia: {}'.format(estimator.inertia_))
	print('Homogeneity: {}'.format(metrics.homogeneity_score(labels, estimator.labels_)))

# clusters = [10, 16, 36, 64, 144, 256, 400]

# # Test different number of clusters
# for n_clusters in clusters:

# 	estimator = MiniBatchKMeans(n_clusters = n_clusters)
# 	estimator.fit(X)

# 	# Print cluster metrics
# 	calculate_metrics(estimator, X, y)

# 	# Determine predicted labels
# 	cluster_labels = infer_cluster_labels(estimator, y)
# 	predicted_y = infer_data_labels(estimator.labels_, cluster_labels)

# 	# Calculate and print accuracy
# 	print('Accuracy: {}\n'.format(metrics.accuracy_score(y, predicted_y)))

# Testing kmeans algorithm on testing dataset
# Convert each image to 1D array
X_test = X_test.reshape(len(X_test), -1)

# Normalize
X_test = X_test.astype(float) / 255.0

# Initialize and fit kmeans algorithm on training data
kmeans = MiniBatchKMeans(n_clusters = 256)
kmeans.fit(X)
cluster_labels = infer_cluster_labels(kmeans, y)

# Predict labels for testing data
test_clusters = kmeans.predict(X_test)
predicted_labels = infer_data_labels(test_clusters, cluster_labels)

# Calculate and print accuracy
print('Testing accuracy: {}'.format(metrics.accuracy_score(y_test, predicted_labels)))

# Visualize cluster centroids

# Initialize and fit kmeans algorithm
kmeans = MiniBatchKMeans(n_clusters = 36)
kmeans.fit(X)

# Record centroid values
centroids = kmeans.cluster_centers_

# Reshape centroids into images
images = centroids.reshape(36, 28, 28)
images *= 255
images = images.astype(np.uint8)

# Determine cluster labels
cluster_labels = infer_cluster_labels(kmeans, y)

# create figure with subplots
fig, axs = plt.subplots(6, 6, figsize = (20, 20))

# Loop through subplots and add centroid images
for i, ax in enumerate(axs.flat):
	# determine inferred labels using cluster_labels dictionary
	for key, value in cluster_labels.items():
		for i in value:
			ax.set_title('inferred_label: {}'.format(key))

	# Add image to subplot
	ax.matshow(images[i])
	ax.axis('off')

# Display the figure
fig.show()
plt.show()