import cv2 as cv

# Print OpenCV Version
print(cv.__version__)

# Read image
image = cv.imread("boy.jpg", 1)

# Convert image to grayscale
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Display image
cv.imshow("Display", image)
cv.waitKey(0)

# Save grayscale image
cv.imwrite("boyGray.jpg",image)
