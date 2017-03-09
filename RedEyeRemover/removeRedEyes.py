'''
    Copyright 2017 by Satya Mallick ( Big Vision LLC )
    http://www.learnopencv.com
'''

import cv2
import numpy as np


def fillHoles(mask):
    '''
        This hole filling algorithm is decribed in this post
        https://www.learnopencv.com/filling-holes-in-an-image-using-opencv-python-c/
    '''
    maskFloodfill = mask.copy()
    h, w = maskFloodfill.shape[:2]
    maskTemp = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(maskFloodfill, maskTemp, (0, 0), 255)
    mask2 = cv2.bitwise_not(maskFloodfill)
    return mask2 | mask

if __name__ == '__main__' :

    # Read image
    img = cv2.imread("red_eyes.jpg", cv2.IMREAD_COLOR)
    
    # Output image
    imgOut = img.copy()
    
    # Load HAAR cascade
    eyesCascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    
    # Detect eyes
    eyes = eyesCascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(100, 100))
    
    # For every detected eye
    for (x, y, w, h) in eyes:

        # Extract eye from the image
        eye = img[y:y+h, x:x+w]

        # Split eye image into 3 channels
        b = eye[:, :, 0]
        g = eye[:, :, 1]
        r = eye[:, :, 2]
        
        # Add the green and blue channels.
        bg = cv2.add(b, g)

        # Simple red eye detector.
        mask = (r > 150) &  (r > bg)
        
        # Convert the mask to uint8 format.
        mask = mask.astype(np.uint8)*255

        # Clean mask -- 1) File holes 2) Dilate (expand) mask.
        mask = fillHoles(mask)
        mask = cv2.dilate(mask, None, anchor=(-1, -1), iterations=3, borderType=1, borderValue=1)

        # Calculate the mean channel by averaging
        # the green and blue channels
        mean = bg / 2
        mask = mask.astype(np.bool)[:, :, np.newaxis]
        mean = mean[:, :, np.newaxis]

        # Copy the eye from the original image.
        eyeOut = eye.copy()

        # Copy the mean image to the output image.
        #np.copyto(eyeOut, mean, where=mask)
        eyeOut = np.where(mask, mean, eyeOut)

        # Copy the fixed eye to the output image.
        imgOut[y:y+h, x:x+w, :] = eyeOut

    # Display Result
    cv2.imshow('Red Eyes', img)
    cv2.imshow('Red Eyes Removed', imgOut)
    cv2.waitKey(0)
