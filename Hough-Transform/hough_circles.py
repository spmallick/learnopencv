import cv2
import numpy as np
import sys

def onTrackbarChange(max_slider):
    cimg = np.copy(img)

    # Detect circles using HoughCircles transform
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, cimg.shape[0]/64, param1=200, param2=10, minRadius=1, maxRadius=max_slider)

    # If at least 1 circle is detected
    if circles is not None:
        cir_len = circles.shape[1] # store length of circles found
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    else:
        cir_len = 0 # no circles detected
    # draw number of circles detected
    cv2.putText(cimg, str(cir_len), (200, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4, cv2.LINE_AA)

    cv2.imshow('Output-Image', cimg)

if __name__ == "__main__":
    # Read image
    img = cv2.imread(sys.argv[1], 1)

    if(img is None):
        print("Image not ready properly.")
        print("Check your file path.")
        sys.exit(0)

    cv2.namedWindow("Output-Image")
    
    # Resize image
    img = cv2.resize(img, (400, 400))

    # Convert to gray-scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise using median blur
    gray_blur = cv2.medianBlur(gray, 3)

    thresh = 10 # initial threshold value for trackbar
    thresh_max = 100 # maximum value for the trackbar for hough transform

    cv2.createTrackbar("threshold", "Output-Image", thresh, thresh_max, onTrackbarChange)
    onTrackbarChange(thresh)

    while True:
        cv2.imshow("source image", img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
