import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers import Conv2D, Input
from keras.optimizers import SGD, Adam
from skimage.metrics import structural_similarity as ssim
import cv2
import os
import math

# Define a function for peak signal-to-noise ratio (PSNR)
def psnr(target, ref):

	# Assume RGB/BGR image
	target_data = target.astype(float)
	ref_data = ref.astype(float)

	diff = ref_data - target_data
	diff = diff.flatten('C')

	rmse = math.sqrt(np.mean(diff ** 2.0))

	# psnr formula
	return 20 * math.log10(255.0 / rmse)

# Define a function for mean squared error
def mse(target, ref):

	err = np.sum((target.astype(float) - ref.astype(float)) ** 2)

	# divide by total number of pixels
	err /= float(target.shape[0] * target.shape[1])

	return err

# Define a function that combines all three of the image quality metrics
def compare_images(target, ref):
	scores = []

	scores.append(psnr(target, ref))
	scores.append(mse(target, ref))
	scores.append(ssim(target, ref, multichannel = True))

	return scores

# Preparing degraded images by introducing quality distortions via resizing
def prepare_images(path, factor):

	# Loop through the files in the directory
	for file in os.listdir(path):

		# open the file
		img = cv2.imread(path + '/' + file)

		# find old and new dimensions
		h, w, c, = img.shape
		new_height = h / factor
		new_width = w / factor
		
		# resize the image - down
		img = cv2.resize(img, (int(new_width), int(new_height)), interpolation = cv2.INTER_LINEAR)

		# resize the image - up
		img = cv2.resize(img, (int(w), int(h)), interpolation = cv2.INTER_LINEAR)

		# Save the image
		print('Saving: {}'.format(file))
		cv2.imwrite('images/{}'.format(file), img)

# Already done
#prepare_images('srcnn', 2)

for file in os.listdir('images/'):

	# Open the target and reference images
	target = cv2.imread('images/{}'.format(file))
	ref = cv2.imread('srcnn/{}'.format(file))

	# Calculate the scores
	scores = compare_images(target, ref)

	# Print the scores
	print('{}\nPSNR: {}\nMSE: {}\nSSIM: {}'.format(file, scores[0], scores[1], scores[2]))

# srcnn model
def model():

	srcnn = Sequential()

	srcnn.add(Conv2D(filters = 128, kernel_size = (9, 9), kernel_initializer = 'glorot_uniform',
		activation = 'relu', padding = 'valid', use_bias = True, input_shape = (None, None, 1)))

	srcnn.add(Conv2D(filters = 64, kernel_size = (3, 3), kernel_initializer = 'glorot_uniform',
		activation = 'relu', padding = 'same', use_bias = True))

	srcnn.add(Conv2D(filters = 1, kernel_size = (5, 5), kernel_initializer = 'glorot_uniform',
		activation = 'linear', padding = 'valid', use_bias = True))

	# Define optimizer
	adam = Adam(learning_rate = 0.0003)

	# Compile model
	srcnn.compile(optimizer = adam, loss = 'mean-squared-error', metrics = ['mean-squared-error'])

	return srcnn


# Weights taken from https://github.com/MarkPrecursor/SRCNN-keras

# Image processing functions
def modcrop(img, scale):

	tmpsz = img.shape
	sz = tmpsz[0:2]
	sz -= np.mod(sz, scale)
	img = img[0:sz[0], 1:sz[1]]

	return img

def shave(image, border):

	img = image[border : -border, border : -border]
	return img

# Define main prediction function
def predict(image_path):

	# Load the srcnn model with weights
	srcnn = model()
	srcnn.load_weights('3051crop_weight_200.h5')

	# Load the degraded and reference images
	path, file = os.path.split(image_path)
	degraded = cv2.imread(image_path)

	ref = cv2.imread('srcnn/{}'.format(file))

	# Preprocess the image with modcrop
	ref = modcrop(ref, 3)
	degraded = modcrop(degraded, 3)

	# Convert the image in YCrCb - (srcnn trained on Y channel)
	temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb)

	# Create image slice and normalize
	Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype = float)
	Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255

	# Perform super-resolution with srcnn
	pre = srcnn.predict(Y, batch_size = 1)

	# Post-process output
	pre *= 255

	pre[pre[:] > 255] = 255
	pre[pre[:] < 0] = 0
	pre = pre.astype(np.uint8)

	# copy Y channel back to image and convert to BGR
	temp = shave(temp, 6)
	temp[:, :, 0] = pre[0, :, :, 0]
	output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

	# Remove border from reference and degraded image
	ref = shave(ref.astype(np.uint8), 6)
	degraded = shave(degraded.astype(np.uint8), 6)

	#image quality calculations
	scores = []
	scores.append(compare_images(degraded, ref))
	scores.append(compare_images(output, ref))

	# return images and scores
	return ref, degraded, output, scores

for file in os.listdir('images'):

	ref, degraded, output, scores = predict('images/{}'.format(file))

	# Print all scores for all images
	print('Degraded Image:\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[0][0], scores[0][1], scores[0][2]))
	print('Reconstructed Image:\nPSNR: {}\nMSE: {}\nSSIM: {}\n'.format(scores[1][0], scores[1][1], scores[1][2]))

	fig, axs = plt.subplots(1, 3, figsize = (20, 8))
	axs[0].imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
	axs[0].set_title('Original')

	axs[1].imshow(cv2.cvtColor(degraded, cv2.COLOR_BGR2RGB))
	axs[1].set_title('Degraded')
	axs[1].set(xlabel = 'PSNR: {}\nMSE: {}\nSSIM: {}'.format(scores[0][0], scores[0][1], scores[0][2]))

	axs[2].imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
	axs[2].set_title('SRCNN')
	axs[2].set(xlabel = 'PSNR: {}\nMSE: {}\nSSIM: {}'.format(scores[1][0], scores[1][1], scores[1][2]))

	for ax in axs:
		ax.set_xticks([])
		ax.set_yticks([])

	print('Saving: '.format(file))

	fig.savefig('output/{}.png'.format(os.path.splitext(file)[0]))

	plt.show()