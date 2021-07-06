import numpy as np
import sklearn
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import datasets

# Load the dataset
iris = datasets.load_iris()
features = iris.data
target = iris.target

# Generate a Pandas DataFrame
df = pd.DataFrame(features)
df.columns = iris.feature_names

# Display scatter plot matrix
scatter_matrix(df)
plt.show()

# Unsupervised learning approach

# Elbow method to determine optimal number of clusters
from sklearn.cluster import KMeans

# Empty X and y lists
X = []
y = []

for i in range(1, 31):
	# Initialize the kmeans model
	kmeans = KMeans(n_clusters = i)
	kmeans.fit(df)

	# Append number of clusters to x data list
	X.append(i)

	# Append average within-cluster sum of squares to y data list
	awcss = kmeans.inertia_ / df.shape[0]
	y.append(awcss)

# Plot the x and y data
plt.plot(X, y, 'bo-')
plt.xlim((1, 30))
plt.xlabel('Number of Clusters')
plt.ylabel('Averageof within-Cluster sum of Squares')
plt.title('KMeans Clustering Elbow Method')

# Display the plot
plt.show()


from sklearn.decomposition import PCA
from sklearn import preprocessing

# Perform PCA
pca = PCA(n_components = 2)
pc = pca.fit_transform(df)

# Print new dimensions
print(pc.shape)
print(pc[:10])

# refit kmeans model to the principle components with the appropriate number of clusters
kmeans = KMeans(n_clusters = 3)
kmeans.fit(pc)

# Visualize high dimensional clusters using principle coponents

# Set size for the mesh
h = 0.02

# Generate mesh grid
x_min, x_max = pc[:, 0].min() - 1, pc[:, 0].max() + 1
y_min, y_max = pc[:, 1].min() - 1, pc[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Label each point in mesh using last trained model
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Generate color plots from results
Z = Z.reshape(xx.shape)
plt.figure(figsize = (12, 12))
plt.clf()
plt.imshow(Z, interpolation = 'nearest', extent = (xx.min(), xx.max(), yy.min(), yy.max()),
			cmap = plt.cm.tab20c, aspect = 'auto', origin = 'lower')

# Plot the principle components on the color plot
for i, point in enumerate(pc):
	if target[i] == 0:
		plt.plot(point[0], point[1], 'g.', markersize = 10)
	if target[i] == 1:
		plt.plot(point[0], point[1], 'b.', markersize = 10)
	if target[i] == 2:
		plt.plot(point[0], point[1], 'r.', markersize = 10)

# Plot the cluster centroids
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], marker = 'x', s = 250, linewidth = 4, color = 'w', zorder = 10)

# Set plot title and axis limits
plt.title('K-Means Clustering on PCA-reduced iris data set')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.xticks(())
plt.yticks(())

plt.show()

# Calculating homogeneity, completeness, and V-measure

# Homogeneity - measures whether or not all of its clusters contain data points which are members of the same class
# Completeness - measures whether or not members of a given class are elements of the same cluster
# V-measure - harmonic mean between homogeneity and completeness

from sklearn import metrics

# Kmeans clustering on non reduced data
kmeans1 = KMeans(n_clusters = 3)
kmeans1.fit(features)

# KMeans Clustering on pca reduced data
kmeans2 = KMeans(n_clusters = 3)
kmeans2.fit(pc)

# Printing metrics for non reduced data
print('Non reduced data')
print('Homogeneity: {}'.format(metrics.homogeneity_score(target, kmeans1.labels_)))
print('Completeness: {}'.format(metrics.completeness_score(target, kmeans1.labels_)))
print('V-measure: {}'.format(metrics.v_measure_score(target, kmeans1.labels_)))

# Printing metrics for non reduced data
print('PCA reduced data')
print('Homogeneity: {}'.format(metrics.homogeneity_score(target, kmeans2.labels_)))
print('Completeness: {}'.format(metrics.completeness_score(target, kmeans2.labels_)))
print('V-measure: {}'.format(metrics.v_measure_score(target, kmeans2.labels_)))

