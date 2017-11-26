import cv2
import numpy as np
import sys

def mouseHandler(event,x,y,flags,param):
    global im_temp, pts_dst

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im_temp,(x,y),3,(0,255,255),5,cv2.LINE_AA)
        cv2.imshow("Image", im_temp)
        if len(pts_dst) < 4:
        	pts_dst = np.append(pts_dst,[(x,y)],axis=0)


# Read in the image.
im_src = cv2.imread(sys.argv[1])
height, width = im_src.shape[:2]

# Create a list of points.
pts_src = np.empty((0,2),dtype=np.int32)
pts_src = np.append(pts_src, [(0,0)], axis=0)
pts_src = np.append(pts_src, [(width-1,0)], axis=0)
pts_src = np.append(pts_src, [(width-1,height-1)], axis=0)
pts_src = np.append(pts_src, [(0,height-1)], axis=0)

# Destination image
im_dst = cv2.imread(sys.argv[2])

# Create a window
cv2.namedWindow("Image", 1)

im_temp = im_dst
pts_dst = np.empty((0,2),dtype=np.int32)

cv2.setMouseCallback("Image",mouseHandler)


cv2.imshow("Image", im_temp)
cv2.waitKey(0)

tform, status = cv2.findHomography(pts_src, pts_dst)
im_temp = cv2.warpPerspective(im_src, tform,(width,height))

cv2.fillConvexPoly(im_dst, pts_dst, 0, cv2.LINE_AA)
im_dst = im_dst + im_temp

cv2.imshow("Image", im_dst)
cv2.waitKey(0)
