#This project describes detecting machine readable zones in passport images
import cv2
import numpy as np
from imutils import contours
from imutils import paths
import imutils
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--images", required=True, help="path to images directory")
args = vars(ap.parse_args())

#initiate a rectangular and square structuring kernel
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13,5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21,21))
    
# loop over the input image paths
for imagePath in paths.list_images(args["images"]):
    #Load the image
    img = cv2.imread(imagePath)
    #img = cv2.resize(img, None, fx=0.3, fy=0.3)

    img = imutils.resize(img, height = 450) #(600, 500, 400, 300)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #imge smoothing using 3x3 Gaussian, then apply the blackhat morphological operator
    #to find dark regions on a light background
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    #compute the Scharr gradient of the blackhat image and scale the result into the range [0,255]
    gradX = cv2.Sobel(blackhat, ddepth = cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    #apply a closing operation using the rectangle kernel to close gaps in between
    #letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    #perform another closing operation, this time using the square kernel to close gaps
    #between lines of the MRZ, then perform a series of erosions to break apart connected
    #components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations = 4)

    #During thresholding, it is possible that border pixels are included in the
    #thresholding, so let's set 5% of the left and right borders to zero
    p = int(img.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, img.shape[1] - p:] = 0

    #Find contours in the threshold image and sort them b their size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)

    #loop over each contour
    for c in cnts:
        #compute bounding box, aspect ratio and coverage ratio of the bounding box
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w/ float(h)
        crWidth = w/ float(gray.shape[1])
        '''print(ar)
        print(crWidth)'''

        #check whether aspect ratio and converage ratio are acceptable
        if ar > 0.5  and crWidth> 0.1:
            #pad bounding box since we applied erosions and now need to regrow it
            pX = int((x + w) * 5)
            pY = int((y + h) * 0.045)
            (x, y) = (x - pX), (y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            #extract ROI from the image and draw a bounding box surrounding it
            roi = img[y : y+ h,x : x + w].copy()
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

    #Rotate the required image
    #img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    #img = cv2.rotate(roi, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

    cv2.imshow("Image", img)
    cv2.imshow("ROI", roi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
