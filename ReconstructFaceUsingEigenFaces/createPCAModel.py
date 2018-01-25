#!/usr/bin/env python3
import os
import sys
import cv2
import numpy as np

# Create data matrix from a list of images
def createDataMatrix(images):
	print("Creating data matrix",end=" ... ", flush=True)
	''' 
	Allocate space for all images in one data matrix.
  The size of the data matrix is
  
  ( w  * h  * 3, numImages )
  
  where,
  
  w = width of an image in the dataset.
  h = height of an image in the dataset.
  3 is for the 3 color channels.
  '''
  
	numImages = len(images)
	sz = images[0].shape
	data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype=np.float32)
	for i in range(0, numImages):
		image = images[i].flatten()
		data[i,:] = image
	
	print("DONE")
	return data

# Read images from the directory
def readImages(path):
	print("Reading images from " + path, end=" ... ", flush=True)
	# Create array of array of images.
	images = []
	# List all files in the directory and read points from text files one by one
	for filePath in sorted(os.listdir(path)):
		fileExt = os.path.splitext(filePath)[1]
		if fileExt in [".jpg", ".jpeg"]:
			# Add to array of images
			imagePath = os.path.join(path, filePath)
			im = cv2.imread(imagePath)

			if im is None :
				print("image:{} not read properly".format(imagePath))
			else :
				# Convert image to floating point
				im = np.float32(im)/255.0
				# Add image to list
				images.append(im)
				# Flip image 
				imFlip = cv2.flip(im, 1);
				# Append flipped image
				images.append(imFlip)

	numImages = len(images) / 2
	# Exit if no image found
	if numImages == 0 :
		print("No images found")
		sys.exit(0)

	print(str(numImages) + " files read.")
	return images



if __name__ == '__main__':

	# Directory containing images
	dirName = "images"

	# Read images
	images = readImages(dirName)
	
	# Size of images
	sz = images[0].shape

	# Create data matrix for PCA.
	data = createDataMatrix(images)

	# Compute the eigenvectors from the stack of images created
	print("Calculating PCA ", end=" ... ", flush=True)
	mean, eigenVectors = cv2.PCACompute(data, mean=None)
	print ("DONE")


	filename = "pcaParams.yml"
	print("Writing size, mean and eigenVectors to " + filename, end=" ... ", flush=True)
	file = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
	file.write("mean", mean)
	file.write("eigenVectors", eigenVectors)
	file.write("size", sz)
	file.release()
	print("DONE")