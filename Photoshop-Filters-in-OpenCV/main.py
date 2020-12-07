import cv2
import numpy as np

def brightness(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('val', 'image', 100, 150, nothing)

    while True:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = np.array(hsv, dtype=np.float64)
        val = cv2.getTrackbarPos('val', 'image')
        val = val/100 # dividing by 100 to get in range 0-1.5

        # scale pixel values up or down for channel 1(Saturation)
        hsv[:, :, 1] = hsv[:, :, 1] * val
        hsv[:, :, 1][hsv[:, :, 1] > 255] = 255 # setting values > 255 to 255.
        # scale pixel values up or down for channel 2(Value)
        hsv[:, :, 2] = hsv[:, :, 2] * val
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255 # setting values > 255 to 255.

        hsv = np.array(hsv, dtype=np.uint8)
        res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow("original", img)
        cv2.imshow('image', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

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

def exponential_function(channel, exp):
    table = np.array([min((i**exp), 255) for i in np.arange(0, 256)]).astype("uint8") # generating table for exponential function
    channel = cv2.LUT(channel, table)
    return channel

def duo_tone(img):
    cv2.namedWindow('image')
    cv2.createTrackbar('exponent', 'image', 0, 10, nothing)
    switch1 = '0 : BLUE n1 : GREEN n2 : RED'
    cv2.createTrackbar(switch1, 'image', 1, 2, nothing)
    switch2 = '0 : BLUE n1 : GREEN n2 : RED n3 : NONE'
    cv2.createTrackbar(switch2, 'image', 3, 3, nothing)
    switch3 = '0 : DARK n1 : LIGHT'
    cv2.createTrackbar(switch3, 'image', 0, 1, nothing)

    while True:
        exp = cv2.getTrackbarPos('exponent', 'image')
        exp = 1 + exp/100 # converting exponent to range 1-2
        s1 = cv2.getTrackbarPos(switch1, 'image')
        s2 = cv2.getTrackbarPos(switch2, 'image')
        s3 = cv2.getTrackbarPos(switch3, 'image')
        res = img.copy()
        for i in range(3):
            if i in (s1, s2): # if channel is present
                res[:, :, i] = exponential_function(res[:, :, i], exp) # increasing the values if channel selected
            else:
                if s3: # for light
                    res[:, :, i] = exponential_function(res[:, :, i], 2 - exp) # reducing value to make the channels light
                else: # for dark
                    res[:, :, i] = 0 # converting the whole channel to 0
        cv2.imshow('Original', img)
        cv2.imshow('image', res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def sepia(img):
    res = img.copy()
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB) # converting to RGB as sepia matrix is for RGB
    res = np.array(res, dtype=np.float64)
    res = cv2.transform(res, np.matrix([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]]))
    res[np.where(res > 255)] = 255 # clipping values greater than 255 to 255
    res = np.array(res, dtype=np.uint8)
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imshow("original", img)
    cv2.imshow("Sepia", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    duo_tone(img)
