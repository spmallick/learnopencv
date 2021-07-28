# Import necessary packages.
from __future__ import print_function
import os
import sys
import cv2
import numpy as np

# Read images from the directory.
def readImages(path):
    print("Reading images from " + path, end = "...")

    # Create array of array of images.
    images = []
    # List all files in the directory and read points from text files one by one.
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:

            # Add to array of images.
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath)

            if im is None :
                print("image:{} not read properly".format(imagePath))
            else :
                # Convert image to floating point.
                im = np.float32(im)/255.0
                # Add image to list.
                images.append(im)
                # Flip image.
                imFlip = cv2.flip(im, 1);
                # Append flipped image.
                images.append(imFlip)
    numImages = int(len(images) / 2)
    # Exit if no image found.
    if numImages == 0 :
        print("No images found")
        sys.exit(0)

    print(str(numImages) + " files read.")
    return images

# Create data matrix from a list of images.
def createDataMatrix(images):
    print("Creating data matrix", end = " ... ")
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
    # Data matrix.
    data = np.zeros((numImages, sz[0] * sz[1] * sz[2]), dtype = np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        # Each row get replaced with one flattened image.
        data[i,:] = image

    print("DONE")
    return data

# Generate new face.
def createNewFace(*args):
    # Start with the mean image.
    output = averageFace

    # Add the eigen faces with the weights.
    for i in range(0, NUM_EIGEN_FACES):
        # Get trackbar position.
        '''
		OpenCV does not allow slider values to be negative. 
		So we use weight = sliderValue - MAX_SLIDER_VALUE / 2
		''' 
        sliderValues[i] = cv2.getTrackbarPos("Weight" + str(i), "Trackbars");
        weight = sliderValues[i] - MAX_SLIDER_VALUE/2
        # Add the weighted eigen face to the mean face.
        output = np.add(output, eigenFaces[i] * weight)

    # Display Result at 2x size.
    output = cv2.resize(output, (0,0), fx = 2, fy = 2)
    cv2.imshow("Result", output)

# Reset sliders callback function.
def resetSliderValues(*args):
    for i in range(0, NUM_EIGEN_FACES):
        cv2.setTrackbarPos("Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2));
    createNewFace()

# Main function.
if __name__ == '__main__':

    # Number of EigenFaces.
    NUM_EIGEN_FACES = 10

    # Maximum weight.
    MAX_SLIDER_VALUE = 255

    # Directory containing images.
    dirName = "images"

    # Read images.
    images = readImages(dirName)

    # Size of images.
    sz = images[0].shape

    # Create data matrix for PCA.
    data = createDataMatrix(images)

    # Compute the eigenvectors from the stack of images created.
    print("Calculating PCA ", end = "...")
    
    mean, eigenVectors = cv2.PCACompute(data, mean = None, maxComponents = NUM_EIGEN_FACES)
    
    print ("DONE")

    averageFace = mean.reshape(sz)

    # Create a container to hold eigen faces.
    eigenFaces  = []

    # Reshape eigen vectors to eigen faces.
    for eigenVector in eigenVectors:
        # REshape.
        eigenFace = eigenVector.reshape(sz)
        # Append eigen faces to the container.
        eigenFaces.append(eigenFace)

    # Create window for displaying result.
    cv2.namedWindow("Result", cv2.WINDOW_NORMAL)
    # Create window for displaying mean face.
    cv2.namedWindow("Average", cv2.WINDOW_NORMAL)

    # Upscale by a factor of two.
    output = cv2.resize(averageFace, (0,0), fx = 2, fy = 2)
    
    # Display.
    cv2.imshow("Result", output)
    cv2.imshow("Average", averageFace)

    # Create Window for trackbars.
    cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

    # Create a list to contain slider values.
    sliderValues = []

    # Create Trackbars.
    for i in range(0, NUM_EIGEN_FACES):
        sliderValues.append(int(MAX_SLIDER_VALUE/2))
        cv2.createTrackbar( "Weight" + str(i), "Trackbars", int(MAX_SLIDER_VALUE/2), MAX_SLIDER_VALUE, createNewFace)

    # You can reset the sliders by clicking on the mean image.
    cv2.setMouseCallback("Average", resetSliderValues);

    print('''Usage:
    Change the weights using the sliders.
    Mouse hover on the result window to reset sliders.
    Press q to terminate.''')

    key = cv2.waitKey(0)
    if key == ord('q'):
        cv2.destroyAllWindows()
