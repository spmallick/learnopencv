import cv2
import numpy as np

num_down = 2
num_bilateral = 7
img = cv2.imread("cybertruck.jpg")        #Taken local image provided in this folder, you can take your own image file

#cv2.imshow("image",img)
#print(img.shape)

img = cv2.resize(img,(800,500))
img_color = img.copy()
gpA = [img_color]

for i in range(num_down):
    img_color = cv2.pyrDown(img_color)
    gpA.append(img_color)
    
for i in range(num_bilateral):
    img_color = cv2.bilateralFilter(img_color, d = 9, sigmaColor = 9, sigmaSpace = 7)
    
for i in range(num_down):
    img_color = cv2.pyrUp(img_color)
    gpA.append(img_color)
    
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.medianBlur(grayscaled, 7)
edged = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
colored = cv2.cvtColor(edged, cv2.COLOR_GRAY2RGB) 

result = cv2.bitwise_and(img, colored)
cv2.imshow("cartoon", result)

stack = np.hstack([img, result])
cv2.imshow('Cartoonized', stack)
cv2.imwrite("Cartoonized.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
