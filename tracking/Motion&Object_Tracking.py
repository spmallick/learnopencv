import cv2
import numpy as np

cap = cv2.VideoCapture('vtest.avi')             #Using here the "vtest.avi" video file from the folder "Videos" of this repo
ret,frame1 = cap.read()
ret,frame2 = cap.read()

fgbg = cv2.createBackgroundSubtractorMOG2()

while True:
    ret,frame = cap.read()
    fgmask = fgbg.apply(frame)

    cv2.imshow('original',frame)
    cv2.imshow('fg',fgmask)
    
    diff = cv2.absdiff(frame1,frame2)
    gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    _,thresh = cv2.threshold(blur,127,255,cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh,None,iterations=2)
    
    contours,hierarchy = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x,y,w,h) = cv2.boundingRect(contour)
        if cv2.contourArea(contour) < 1000:
            continue
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame1,"Object:Moving{}".format('Movement'),(10,20),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
    
    #cv2.drawContours(frame1,contours,-1,(0,255,0),2)
    cv2.imshow('Input',frame1)
    frame1 = frame2
    ret,frame2 = cap.read()
    if cv2.waitKey(40) == 27:
        break
        
cv2.destroyAllWindows()
cap.release()
