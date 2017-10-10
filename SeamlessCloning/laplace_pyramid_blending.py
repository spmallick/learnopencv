import cv2
import numpy as np

# Read two images
A = cv2.imread('images/apple.png')
B = cv2.imread('images/orange.jpg')

# Start with original images (base of pyramids)
gaussianA = A.copy()
gaussianB = B.copy()

# combined laplacian pyramids of both images
combinedLaplacianPyramids = []

# Number of levels in pyramids, try with different values 
maxIterations = 6

for i in range(maxIterations):

	# compute laplacian pyramids for both images
	laplacianA = cv2.subtract(gaussianA, cv2.pyrUp(cv2.pyrDown(gaussianA)))
	laplacianB = cv2.subtract(gaussianB, cv2.pyrUp(cv2.pyrDown(gaussianB)))

	# stack their halves together (left half of A + right half of B)
	width = laplacianA.shape[1]
	combinedLaplacian = np.hstack((laplacianA[:,0:width/2], laplacianB[:,width/2:]))
	
	# add combinedLaplacian in the beginning of the list of combined laplacian pyramids
	combinedLaplacianPyramids.insert(0,combinedLaplacian)

	# Update guassian pyramids for next iteration
	gaussianA = cv2.pyrDown(gaussianA)
	gaussianB = cv2.pyrDown(gaussianB)

# Add last combination of laplacian pyramids (top level of pyramids)
width = gaussianA.shape[1]
lastCombined = np.hstack((gaussianA[:,0:width/2], gaussianB[:,width/2:]))
combinedLaplacianPyramids.insert(0,lastCombined)

# reconstructing image
blendedImage = combinedLaplacianPyramids[0]
for i in xrange(1,len(combinedLaplacianPyramids)):
    blendedImage = cv2.pyrUp(blendedImage)
    blendedImage = cv2.add(blendedImage, combinedLaplacianPyramids[i])

# direct combining both halves for comparison
width = A.shape[1]
directCombination = np.hstack((A[:,:width/2],B[:,width/2:]))

cv2.imshow('Blended',blendedImage)
cv2.imshow('Direct combination',directCombination)
cv2.waitKey(0)