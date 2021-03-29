import cv2

# read the image
image = cv2.imread('../../input/image_1.jpg')
# convert the image to grayscale format
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
# visualize the binary image
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('image_thres1.jpg', thresh)
cv2.destroyAllWindows()

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, 
                                       method=cv2.CHAIN_APPROX_NONE)
# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, 
                 color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('contours_none_image1.jpg', image_copy)
cv2.destroyAllWindows()

"""
Now let's try with `cv2.CHAIN_APPROX_SIMPLE`
"""
# detect the contours on the binary image using cv2.ChAIN_APPROX_SIMPLE
contours1, hierarchy1 = cv2.findContours(thresh, cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE)
# draw contours on the original image for `CHAIN_APPROX_SIMPLE`
image_copy1 = image.copy()
cv2.drawContours(image_copy1, contours1, -1, (0, 255, 0), 2, 
                 cv2.LINE_AA)
# see the results
cv2.imshow('Simple approximation', image_copy1)
cv2.waitKey(0)
cv2.imwrite('contours_simple_image1.jpg', image_copy1)
cv2.destroyAllWindows()

# to actually visualize the effect of `CHAIN_APPROX_SIMPLE`, we need a proper image
image1 = cv2.imread('../../input/image_2.jpg')
img_gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
ret, thresh1 = cv2.threshold(img_gray1, 150, 255, cv2.THRESH_BINARY)
contours2, hierarchy2 = cv2.findContours(thresh1, cv2.RETR_TREE, 
                                                cv2.CHAIN_APPROX_SIMPLE)
image_copy2 = image1.copy()
cv2.drawContours(image_copy2, contours2, -1, (0, 255, 0), 2, 
                 cv2.LINE_AA)
cv2.imshow('SIMPLE Approximation contours', image_copy2)
cv2.waitKey(0)
image_copy3 = image1.copy()
for i, contour in enumerate(contours2): # loop over one contour area
    for j, contour_point in enumerate(contour): # loop over the points 
        # draw a circle on the current contour coordinate
        cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 
                   2, (0, 255, 0), 2, cv2.LINE_AA)
# see the results
cv2.imshow('CHAIN_APPROX_SIMPLE Point only', image_copy3)
cv2.waitKey(0)
cv2.imwrite('contour_point_simple.jpg', image_copy3)
cv2.destroyAllWindows()