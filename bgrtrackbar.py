# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 21:04:49 2018

@author: charc
"""

import cv2
import numpy as np
     
def nothing(x):
    pass
     
     
img1 = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('BGR TRACKBAR')
    
cv2.createTrackbar('R','BGR TRACKBAR',0,255, nothing)
cv2.createTrackbar('G','BGR TRACKBAR',0,255, nothing)
cv2.createTrackbar('B','BGR TRACKBAR',0,255, nothing)
   
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'BGR TRACKBAR', 0, 1, nothing)
    
while(1):
    cv2.imshow('BGR TRACKBAR',img1)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    
    r = cv2.getTrackbarPos('R','BGR TRACKBAR')
    g = cv2.getTrackbarPos('G','BGR TRACKBAR')
    b = cv2.getTrackbarPos('B','BGR TRACKBAR')
    s = cv2.getTrackbarPos(switch,'BGR TRACKBAR')
   
    if s == 0:
        img1[:] = 0
    else:
        img1[:] = [b,g,r]
        print(b, g, r)
   

cv2.destroyAllWindows()