'''
OpenCV Non-Photorealistic Rendering Python Example

Copyright 2015 by Satya Mallick <spmallick@gmail.com>
'''

import cv2

# Read image
im = cv2.imread("cow.jpg");

# Edge preserving filter with two different flags.
imout = cv2.edgePreservingFilter(im, flags=cv2.RECURS_FILTER);
cv2.imwrite("edge-preserving-recursive-filter.jpg", imout);

imout = cv2.edgePreservingFilter(im, flags=cv2.NORMCONV_FILTER);
cv2.imwrite("edge-preserving-normalized-convolution-filter.jpg", imout);

# Detail enhance filter
imout = cv2.detailEnhance(im);
cv2.imwrite("detail-enhance.jpg", imout);

# Pencil sketch filter
imout_gray, imout = cv2.pencilSketch(im, sigma_s=60, sigma_r=0.07, shade_factor=0.05);
cv2.imwrite("pencil-sketch.jpg", imout_gray);
cv2.imwrite("pencil-sketch-color.jpg", imout);

# Stylization filter
cv2.stylization(im,imout);
cv2.imwrite("stylization.jpg", imout);

