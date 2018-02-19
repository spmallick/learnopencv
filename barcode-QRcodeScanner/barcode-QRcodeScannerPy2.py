from __future__ import print_function
from sys import argv
import zbar
import cv2
import numpy as np

# Find barcodes and QR codes
def decode(im):
	
	# Create zbar scanner
	scanner = zbar.ImageScanner()
	
	# Configure scanner
	scanner.parse_config('enable')	

	# Convert image to grayscale
	imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

	# Find height and width
	height, width = imGray.shape

	# Get raw image bytes
	raw = imGray.tobytes()

	# Wrap image data in a zbar image
	decodedObjects = zbar.Image(width, height, 'Y800', raw)

	# Scan the image for barcodes and QRCodes
	scanner.scan(decodedObjects)

	# Print results 
	for decodedObject in decodedObjects:
		print('Type : ', decodedObject.type); 
		print('Data :', decodedObject.data,'\n'); 

	# Return decoded object
	return decodedObjects
		
# Display barcode and QR code location	
def display(im, decodedObjects):

	# Loop over all decoded objects
	for decodedObject in decodedObjects: 
		points = decodedObject.location
    
    # If the points do not form a quad, find convex hull
		if len(points) > 4 : 
			hull = cv2.convexHull(np.array([point for point in points]))
			hull = map(tuple, np.squeeze(hull))
		else : 
			hull = points;
    
    # Number of points in the convex hull
		n = len(hull)

    # Draw the convext hull
		for j in xrange(0,n):
			cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)

	# Display results 
	cv2.imshow("Results", im);
	cv2.waitKey(0);
  

# Main 
if __name__ == '__main__':

	# Read image using OpenCV 
	im = cv2.imread('zbar-test.jpg')
	
	# Decode Image
	decodedObjects = decode(im)

	# Display location 
	display(im, decodedObjects)

	









