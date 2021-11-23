import cv2
import sys
import time


def displayBbox(im, bbox):
    if bbox is not None:
        bbox = [bbox[0].astype(int)]
        n = len(bbox[0])
        for i in range(n):
            cv2.line(im, tuple(bbox[0][i]), tuple(bbox[0][(i+1) % n]), (0,255,0), 3)


# INITIALIZATION.
# Instantiate QR Code detector object.
detector = cv2.wechat_qrcode_WeChatQRCode("../model/detect.prototxt",
    "../model/detect.caffemodel",
    "../model/sr.prototxt",
    "../model/sr.caffemodel")


if __name__ == '__main__':
    # Load image.
    if len(sys.argv)>1:
        img = cv2.imread(sys.argv[1])
    else:
        img = cv2.imread('sample-qrcode.jpg')

    t1 = time.time()
    # Detect and decode.
    res, points = detector.detectAndDecode(img)
    t2 = time.time()
    # Detected outputs.
    if len(res) > 0:
        print('Time Taken : ', round(1000*(t2 - t1),1), ' ms')
        print('Output : ', res[0])
        print('Bounding Box : ', points)
        displayBbox(img, points)
    else:
        print('QRCode not detected')

    cv2.imshow("Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()