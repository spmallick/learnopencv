from pyimagesearch.transform import four_point_transform
from skimage.filters import threshold_local
import numpy as np
import cv2
import argparse
import imutils

#construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image",required = True,help="Path to the image to be scanned")
args = vars(ap.parse_args())
image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image,height=500)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),0)
edged = cv2.Canny(gray,75,200)
print("STEP 1:Edge detection")
#cv2.imshow("Image",image)
#cv2.imshow("Edged",edged)

cnts = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts,key=cv2.contourArea,reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c,True)
    approx = cv2.approxPolyDP(c,0.02*peri,True)
    #if contour hass four points,we have found our screen
    if len(approx) == 4:
        screenCnt = approx
        break

print("STEP 2:Find contours")
cv2.drawContours(image,[screenCnt],-1,(0,255,0),2)
cv2.imshow("Outline",image)

#apply four_point_transform to get top-down view of the image
warped = four_point_transform(orig,screenCnt.reshape(4,2)*ratio)
warped = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
T = threshold_local(warped,11,offset=10,method='gaussian')
warped = (warped > T).astype("uint8")*255

print("STEP 3:Apply perspective transform")
cv2.imshow("Original",imutils.resize(orig,height=200))
cv2.imshow("Scanned Image",imutils.resize(warped,height=400))
cv2.waitKey(0)
cv2.destroyAllWindows()
