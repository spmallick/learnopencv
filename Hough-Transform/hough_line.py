import cv2
import numpy as np
import sys

def onTrackbarChange(max_slider):
	global img
	global dst
	global gray

	dst = np.copy(img)
	# Detect edges in the image
	edges = cv2.Canny(gray, 50, 200)
	# Apply hough line transform
	lines = cv2.HoughLinesP(edges, 1, np.pi/180, max_slider, minLineLength=10, maxLineGap=250)

	# Draw lines on the detected points
	for line in lines:
		x1, y1, x2, y2 = line[0]
		cv2.line(dst, (x1, y1), (x2, y2), (255,0,0), 3)

	cv2.imshow("Result Image", dst)	

if __name__ == "__main__":
	if(len(sys.argv) > 1):
			# Read image
			img = cv2.imread(sys.argv[1], 1)
	else:
		print("Path not specified. Please specify image path")
		sys.exit(0)
	
	# Create a copy for later usage
	dst = np.copy(img)

	cv2.namedWindow("Result Image")

	# Convert image to gray
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Initialize thresh value
	thresh = 80

	# max_value of thresh
	val = 100

	cv2.createTrackbar("threshold", "Result Image", thresh, val, onTrackbarChange)
	onTrackbarChange(val)

	while True:
		cv2.imshow("Source Image",img)
		key = cv2.waitKey(1)
		if key == ord('q'):
			break

	cv2.destroyAllWindows()
