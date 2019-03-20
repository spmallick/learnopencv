import cv2
import numpy as np
import sys

def onTrackbarChange(max_slider):
    cimg = np.copy(img)

    p1 = max_slider
    p2 = max_slider * 0.4

    # Detect circles using HoughCircles transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, cimg.shape[0]/64, param1=p1, param2=p2, minRadius=25, maxRadius=50)

    # If at least 1 circle is detected
    if circles is not None:
        cir_len = circles.shape[1] # store length of circles found
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        cir_len = 0 # no circles detected
    
    # Display output image
    cv2.imshow('Image', cimg)    

    # Edge image for debugging
    edges = cv2.Canny(gray, p1, p2)
    cv2.imshow('Edges', edges)

    

    
if __name__ == "__main__":
    # Read image
    img = cv2.imread(sys.argv[1], 1)

    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create display windows
    cv2.namedWindow("Edges")
    cv2.namedWindow("Image")
    

    # Trackbar will be used for changing threshold for edge 
    initThresh = 105 
    maxThresh = 200 

    # Create trackbar
    cv2.createTrackbar("Threshold", "Image", initThresh, maxThresh, onTrackbarChange)
    onTrackbarChange(initThresh)
    
    while True:
        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
