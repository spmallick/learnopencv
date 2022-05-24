#!/usr/bin/env python

'''
    OpenCV Colormap  Example
    
    Copyright 2015 by Satya Mallick <spmallick@learnopencv.com>
    
'''


import cv2
import numpy as np


def colormap_name(id) :
    switcher = {
        0 : "COLORMAP_AUTUMN",
        1 : "COLORMAP_BONE",
        2 : "COLORMAP_JET",
        3 : "COLORMAP_WINTER",
        4 : "COLORMAP_RAINBOW",
        5 : "COLORMAP_OCEAN",
        6 : "COLORMAP_SUMMER",
        7 : "COLORMAP_SPRING",
        8 : "COLORMAP_COOL",
        9 : "COLORMAP_HSV",
        10: "COLORMAP_PINK",
        11: "COLORMAP_HOT"
        
    }
    return switcher.get(id, "NONE")


if __name__ == '__main__' :

    im = cv2.imread("pluto.jpg", cv2.IMREAD_GRAYSCALE)
    im_out = np.zeros((600, 800, 3), np.uint8);

    for i in xrange(0,4) :
        for j in xrange(0,3) :
            k = i + j * 4
            im_color = cv2.applyColorMap(im, k)
            cv2.putText(im_color, colormap_name(k), (30, 180), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.CV_AA);
            
            ix200 = i * 200
            jx200 = j * 200
            
            im_out[ jx200 : jx200 + 200 , ix200 : ix200 + 200 , : ] = im_color

    cv2.imshow("Pseudo Colored", im_out);
    cv2.waitKey(0);
