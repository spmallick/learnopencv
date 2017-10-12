import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read two images
A = cv2.imread("images/man.jpg")
B = cv2.imread("images/woman.jpg")

# Convert to float
A = np.float32(A) / 255.0
B = np.float32(B) / 255.0

# Create a rough mask around man face in A.
mask = np.zeros(A.shape,A.dtype)
polygon = np.array([[164,226], [209,225], [238,188], [252,133], [248,75], [240,29], [192,15], [150,15], [100,70], [106,133], [123,194] ], np.int32)
cv2.fillPoly(mask, [polygon], (255, 255, 255))
# Convert the mask to float
mask = np.float32(mask) / 255.0

# Multiply with float < 1.0 to take weighted average of man and woman's face
mask = mask * 0.7 # 0.7 for man, 0.3 for woman

# Resizing to multiples of 2^(levels in pyramid), thus 32 in our case
A = cv2.resize(A,(384,352))

# B and mask should have same size as A for multiplication and addition operations later
B = cv2.resize(B,(A.shape[1],A.shape[0]))
mask = cv2.resize(mask,(A.shape[1],A.shape[0]))

# Start with original images (base of pyramids)
guassianA = A.copy()
guassianB = B.copy()
guassianMask = mask.copy()

# combined laplacian pyramids of both images
combinedLaplacianPyramids = []

# Number of levels in pyramids, try with different values, Be careful with image sizes
maxIterations = 5

for i in range(maxIterations):

	# compute laplacian pyramids for both images
	laplacianA = cv2.subtract(guassianA, cv2.pyrUp(cv2.pyrDown(guassianA)))
	laplacianB = cv2.subtract(guassianB, cv2.pyrUp(cv2.pyrDown(guassianB)))

	# Combine both laplacian pyramids, taking weighted average with guassian pyramid of mask
	combinedLaplacian = guassianMask * laplacianA + (1.0 - guassianMask) * laplacianB

	# add combinedLaplacian in the beginning of the list of combined laplacian pyramids
	combinedLaplacianPyramids.insert(0,combinedLaplacian)

	# Update guassian pyramids for next iteration
	guassianA = cv2.pyrDown(guassianA)
	guassianB = cv2.pyrDown(guassianB)
	guassianMask = cv2.pyrDown(guassianMask)

# Add last combination of laplacian pyramids (top level of pyramids)
lastCombined = guassianMask * guassianA + (1.0 - guassianMask) * guassianB
combinedLaplacianPyramids.insert(0,lastCombined)

# reconstructing image
blendedImage = combinedLaplacianPyramids[0]
for i in xrange(1,len(combinedLaplacianPyramids)):
    # upSample and add to next level
    blendedImage = cv2.pyrUp(blendedImage)
    blendedImage = cv2.add(blendedImage, combinedLaplacianPyramids[i])

cv2.imshow('Blended',blendedImage)

# direct blending both images for comparison
directCombination = mask * A + (1.0 - mask) * B
cv2.imshow('Direct combination',directCombination)

cv2.waitKey(0)