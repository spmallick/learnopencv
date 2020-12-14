import cv2
import numpy as np

def nothing(x):
    pass
    
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
    
if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    duo_tone(img)
