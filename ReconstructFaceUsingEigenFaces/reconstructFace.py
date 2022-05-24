# Import necessary packages
import os
import sys
import cv2
import numpy as np

'''
 Display result
 Left = Original Image
 Right = Reconstructed Face
'''
def displayResult(left, right)	:
	output = np.hstack((left,right))	
	output = cv2.resize(output, (0,0), fx=4, fy=4)
	cv2.imshow("Result", output)

# Recontruct face using mean face and EigenFaces
def reconstructFace(*args):
	# Start with the mean / average face
	output = averageFace
	
	for i in range(0,args[0]):
		'''
		The weight is the dot product of the mean subtracted
		image vector with the EigenVector
		'''
		weight = np.dot(imVector, eigenVectors[i])
		output = output + eigenFaces[i] * weight

	
	displayResult(im, output)
    


if __name__ == '__main__':

	# Read model file
	modelFile = "pcaParams.yml"
	print("Reading model file " + modelFile, end=" ... ", flush=True)
	file = cv2.FileStorage(modelFile, cv2.FILE_STORAGE_READ)
	
	# Extract mean vector
	mean = file.getNode("mean").mat()
	
	# Extract Eigen Vectors
	eigenVectors = file.getNode("eigenVectors").mat()
	
	# Extract size of the images used in training.
	sz = file.getNode("size").mat()
	sz = (int(sz[0,0]), int(sz[1,0]), int(sz[2,0]))
	
	''' 
	Extract maximum number of EigenVectors. 
	This is the max(numImagesUsedInTraining, w * h * 3)
	where w = width, h = height of the training images. 
	'''

	numEigenFaces = eigenVectors.shape[0]
	print("DONE")

	# Extract mean vector and reshape it to obtain average face
	averageFace = mean.reshape(sz)

	# Reshape Eigenvectors to obtain EigenFaces
	eigenFaces = [] 
	for eigenVector in eigenVectors:
		eigenFace = eigenVector.reshape(sz)
		eigenFaces.append(eigenFace)


	# Read new test image. This image was not used in traning. 
	imageFilename = "test/satya2.jpg"
	print("Read image " + imageFilename + " and vectorize ", end=" ... ");
	im = cv2.imread(imageFilename)
	im = np.float32(im)/255.0

	# Reshape image to one long vector and subtract the mean vector
	imVector = im.flatten() - mean; 
	print("Done");
	
	# Show mean face first
	output = averageFace
	
	# Create window for displaying result
	cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)

	# Changing the slider value changes the number of EigenVectors
	# used in reconstructFace.
	cv2.createTrackbar( "No. of EigenFaces", "Result", 0, numEigenFaces, reconstructFace)

	# Display original image and the reconstructed image size by side
	displayResult(im, output)

	cv2.waitKey(0)
	cv2.destroyAllWindows()
