import cv2
import numpy as np

def nothing(x):
    pass

def tv_60(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('val', 'image', 0, 255, nothing)
    cv2.createTrackbar('threshold', 'image', 0, 100, nothing)

    while True:
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.getTrackbarPos('threshold', 'image')
        val = cv2.getTrackbarPos('val', 'image')
        for i in range(height):
            for j in range(width):
                if np.random.randint(100) <= thresh:
                    if np.random.randint(2) == 0:
                        gray[i, j] = min(gray[i, j] + np.random.randint(0, val+1), 255) # adding noise to image and setting values > 255 to 255. 
                    else:
                        gray[i, j] = max(gray[i, j] - np.random.randint(0, val+1), 0) # subtracting noise to image and setting values < 0 to 0.

        cv2.imshow('Original', img)
        cv2.imshow('image', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    tv_60(img)
