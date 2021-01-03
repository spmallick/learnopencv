import cv2
import numpy as np

def nothing(x):
    pass
 

def kernel_generator(size):
    kernel = np.zeros((size, size), dtype=np.int8)
    for i in range(size):
        for j in range(size):
            if i < j:
                kernel[i][j] = -1
            elif i > j:
                kernel[i][j] = 1
    return kernel

def emboss(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('size', 'image', 0, 8, nothing)
    switch = '0 : BL n1 : BR n2 : TR n3 : BR'
    cv2.createTrackbar(switch, 'image', 0, 3, nothing)

    while True:
        size = cv2.getTrackbarPos('size', 'image')
        size += 2 # adding 2 to kernel as it a size of 2 is the minimum required.
        s = cv2.getTrackbarPos(switch, 'image')
        height, width = img.shape[:2]
        y = np.ones((height, width), np.uint8) * 128
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = kernel_generator(size) # generating kernel for bottom left kernel
        kernel = np.rot90(kernel, s) # switching kernel according to direction
        res = cv2.add(cv2.filter2D(gray, -1, kernel), y)

        cv2.imshow('Original', img)
        cv2.imshow('image', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    emboss(img)
