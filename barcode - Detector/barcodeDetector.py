import numpy as np
import argparse
import imutils
import cv2

#create argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,help = "path to the image file")
args = vars(ap.parse_args())

#load the image and convert to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#compute scharr gradient of the image
ddepth = cv2.cv.CV_32F if imutils.is_cv2() else cv2.CV_32F
gradX = cv2.Sobel(gray,ddepth=ddepth,dx=1,dy=0,ksize=-1)
gradY = cv2.Sobel(gray,ddepth=ddepth,dx=0,dy=1,ksize=-1)

#subtract y-gradient from x-gradient
gradient = cv2.subtract(gradX,gradY)
gradient = cv2.convertScaleAbs(gradient)

#blur and threshold image
blurred = cv2.blur(gradient,(9,9))
(_,thresh) = cv2.threshold(blurred,255,255,cv2.THRESH_BINARY)

#construct closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(21,7))
closed = cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
closed = cv2.erode(closed,None,iterations = 4)
closed = cv2.dilate(closed,None,iterations = 4)

#find and approximate the contours
cnts = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = sorted(cnts,key = cv2.contourArea,reverse = True)[1]
rect = cv2.minAreaRect(c)
box = cv2.cv.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect)
box = np.int0(box)

#draw the image
cv2.drawContours(image,[box],-1,(0,255,0),3)
cv2.imshow("Image",image)
cv2.waitKey(0)
