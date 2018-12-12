import cv2
import numpy as np
import sys
import time
import pyzbar.pyzbar as pyzbar

cap = cv2.VideoCapture(0)
hasFrame,frame = cap.read()
vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame.shape[1],frame.shape[0]))

# Display barcode and QR code location
def display(im, decodedObjects):

  # Loop over all decoded objects
  for decodedObject in decodedObjects:
    points = decodedObject.polygon

    # If the points do not form a quad, find convex hull
    if len(points) > 4 :
      hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
      hull = list(map(tuple, np.squeeze(hull)))
    else :
      hull = points;

    # Number of points in the convex hull
    n = len(hull)

    # Draw the convext hull
    for j in range(0,n):
      cv2.line(im, hull[j], hull[ (j+1) % n], (255,0,0), 3)

  # Display results
  # cv2.imshow("Results", im);

# Create a qrCodeDetector Object
qrDecoder = cv2.QRCodeDetector()

# Detect and decode the qrcode
t = time.time()
while(1):
    hasFrame, inputImage = cap.read()
    if not hasFrame:
        break
    decodedObjects = pyzbar.decode(inputImage)
    if len(decodedObjects):
        zbarData = decodedObjects[0].data
    else:
        zbarData=''
    opencvData,bbox,rectifiedImage = qrDecoder.detectAndDecode(inputImage)
    if zbarData:
        cv2.putText(inputImage, "ZBAR : {}".format(zbarData), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(inputImage, "ZBAR : QR Code NOT Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    if opencvData:
        cv2.putText(inputImage, "OpenCV:{}".format(opencvData), (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(inputImage, "OpenCV:QR Code NOT Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    display(inputImage, decodedObjects)
    cv2.imshow("Result",inputImage)
    vid_writer.write(inputImage)
    k = cv2.waitKey(20)
    if k == 27:
        break
cv2.destroyAllWindows()
vid_writer.release()
