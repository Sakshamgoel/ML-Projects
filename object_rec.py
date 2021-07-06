import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from keras.datasets import cifar10
from keras.utils import np_utils
from PIL import Image


# # Load the data
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# # Lets determine the dataset characteristics
# print('Training images: {}'.format(X_train.shape))
# print('Testing images: {}'.format(X_test.shape))
# print('Size of one image: {}'.format(X_train[0].shape))

# # Grid of 3*3 images
# for i in range(0, 9):
# 	plt.subplot(330 + 1 + i)
# 	img = X_train[50 + i]
# 	plt.imshow(img)

# # Showing the plot
# plt.show()

# Preprocessing the dataset

# Fix random seed for reproducibility
seed = 6
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Normalizing the input from 0-225 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

#print(X_train[0])

# Hot encoding the output
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_class = y_test.shape[1]

from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD

# Define a model function
def allcnn(weight = None):

	# Defining the model type
	model = Sequential()

	# Adding model layers
	model.add(Conv2D(96, (3, 3), padding = 'same', input_shape = (32, 32, 3)))
	model.add(Activation('relu'))
	model.add(Conv2D(96, (3, 3), padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(192, (3, 3), padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(192, (3, 3), padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2, 2)))
	model.add(Dropout(0.5))

	model.add(Conv2D(192, (3, 3), padding = 'same'))
	model.add(Activation('relu'))
	model.add(Conv2D(192, (1, 1), padding = 'valid'))
	model.add(Activation('relu'))
	model.add(Conv2D(10, (1, 1), padding = 'valid'))

	# Add Global Average Pooling Layer with softmax Activation
	model.add(GlobalAveragePooling2D())
	model.add(Activation('softmax'))

	# Load the weights
	if weight:
		model.load_weights(weights)


	return model


# Hyper parameters
learning_rate = 0.01
weight_decay = 1e-6
momentum = 0.9

# Build model
weights = 'all_cnn_weights_0.9088_0.4994.hdf5'
model = allcnn(weights)

# Defining an optimizer and compiling the model
sgd = SGD(learning_rate = learning_rate, decay = weight_decay, momentum = momentum, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

#print(model.summary())

# Defining additional training parameters
epochs = 350
batch_size = 32

# # Fit the model
# model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochs, batch_size = batch_size, verbose = 1)

# Testing the model with pretrained weights
scores = model.evaluate(X_test, y_test, verbose = 1)
print('Accuracy: {}'.format(scores[1]))

classes = range(0, 10)

names = [
	'airplane',
	'automobile',
	'bird',
	'cat',
	'deer',
	'dog',
	'frog',
	'horse',
	'ship',
	'truck'
]

# zip the names and classes together
class_labels = dict(zip(classes, names))
print(class_labels)

# Generate batch of 9 images to predict
batch = X_test[100:109]
labels = np.argmax(y_test[100:109], axis = -1)

# Make the predictions
predictions = model.predict(batch, verbose = 1)

# Print out the predictions
print(predictions)

class_results = np.argmax(predictions, axis = -1)
print(class_results)

# Create a grid of 3 x 3 images
fig, axs = plt.subplots(3, 3,figsize = (15, 6))
fig.subplots_adjust(hspace = 1)
axs = axs.flatten()

for i, img in enumerate(batch):

	# Start by determining label
	for key, value in class_labels.items():
		if class_results[i] == key:
			title = 'Prediction: {} \nActual: {}'.format(class_labels[key], class_labels[labels[i]])
			axs[i].set_title(title)
			axs[i].axes.get_xaxis().set_visible(False)
			axs[i].axes.get_yaxis().set_visible(False)

	# Plot the image
	axs[i].imshow(img)

# Show the plot
plt.show()
