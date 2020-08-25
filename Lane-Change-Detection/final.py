import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

NORM_FONT= ("Verdana", 10)

def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("Message")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()

cascade_src = 'cascade/cars.xml'
video_src = 'dataset/cars.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while(1):
    # Take frame by frame
    _, frame = cap.read()
    cars = car_cascade.detectMultiScale(frame, 1.1, 1)
    for (x,y,w,h) in cars:
        roi = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)   #ROI is region of interest
    #canceling noise in the video frames using blur
    frame = cv2.GaussianBlur(frame,(21,21),0)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of color in HSV to see the points at which the car is changing angles
    lower_limit = np.array([0,150,150])
    upper_limit = np.array([10,255,255])

    # Threshold the HSV image to get only the thresholded colors
    mask = cv2.inRange(hsv, lower_limit, upper_limit)

    kernel = np.ones((3,3),np.uint8)
    kernel_lg = np.ones((15,15),np.uint8)
    
    # image processing technique called the erosion is used for noise reduction
    mask = cv2.erode(mask,kernel,iterations = 1)

    # image processing technique called the dilation is used to regain some lost area
    mask = cv2.dilate(mask,kernel_lg,iterations = 1)    
    
    # Bitwise-AND to get black everywhere else except the region of interest
    result = cv2.bitwise_and(frame,frame, mask= mask)
    
    thresh = mask
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    
    # define a minimum area for a contour (ignoring all values below min)
    min_area = 1000
    cont_filtered = []    
    
    # filter out all contours below a min_area
    for cont in contours:
        if cv2.contourArea(cont) > min_area:
            cont_filtered.append(cont)
    try:
        cnt = cont_filtered[0]
        
        # draw the rectangles around contours
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(frame,[box],0,(0,0,255),2)
        
        rows,cols = thresh.shape[:2]
        [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
        lefty = int((-x*vy/vx) + y)
        righty = int(((cols-x)*vy/vx)+y)
        cv2.line(frame,(cols-1,righty),(0,lefty),(0,255,0),2)        
        
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        (x,y),(MA,ma),angle = cv2.fitEllipse(cnt)
        print('x= ', cx, '  y= ', cy, ' angle = ', round(rect[2],2))
        if(round(rect[2],2))<-45:
            # print('Lane change detected')
            popupmsg('Lane change detected')

    except:
        print('no contours')

    cv2.imshow('frame',frame)
    cv2.imshow('result', result)
    
    k = cv2.waitKey(5) & 0xFF
    if k == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()