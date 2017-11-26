import cv2
import numpy as np

def mouseHandler(event,x,y,flags,param):
    global im_temp, pts_src

    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(im_temp,(x,y),3,(0,255,255),5,cv2.LINE_AA)
        cv2.imshow("Image", im_temp)
        if len(pts_src) < 4:
        	pts_src = np.append(pts_src,[(x,y)],axis=0)


# Read in the image.
im_src = cv2.imread("book.jpg")

# Destination image
height, width = 400, 300
im_dst = np.zeros((height,width,3),dtype=np.uint8)


# Create a list of points.
pts_dst = np.empty((0,2))
pts_dst = np.append(pts_dst, [(0,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,0)], axis=0)
pts_dst = np.append(pts_dst, [(width-1,height-1)], axis=0)
pts_dst = np.append(pts_dst, [(0,height-1)], axis=0)

# Create a window
cv2.namedWindow("Image", 1)

im_temp = im_src
pts_src = np.empty((0,2))

cv2.setMouseCallback("Image",mouseHandler)


cv2.imshow("Image", im_temp)
cv2.waitKey(0)

tform, status = cv2.findHomography(pts_src, pts_dst)
im_dst = cv2.warpPerspective(im_src, tform,(width,height))

cv2.imshow("Image", im_dst)
cv2.waitKey(0)
