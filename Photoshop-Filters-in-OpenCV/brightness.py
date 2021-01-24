import cv2
import numpy as np

def nothing(x):
    pass

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
    
if __name__ == "__main__":
    img = cv2.imread("image.jpg")
    brightness(img)
