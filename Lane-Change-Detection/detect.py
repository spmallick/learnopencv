import cv2
import numpy as np

cascade_src = 'cascade/cars.xml'
video_src = 'dataset/cars.mp4'

cap = cv2.VideoCapture(video_src)
car_cascade = cv2.CascadeClassifier(cascade_src)

while True:
    ret, frame = cap.read()
    if (type(frame) == type(None)):
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    for (x,y,w,h) in cars:
        roi_gray = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)   #ROI is region of interest
        img_item = "1.png"
        cv2.imwrite(img_item, roi_gray)

    #lane detection
    def canny(frame):
        gray=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        canny=cv2.Canny(blur,50,150)  
        return canny
    def region_of_interest(frame):
        height=frame.shape[0]
        polygons=np.array([
                          [(0,height),(500,0),(800,0),(1300,550),(1100,height)]
                           ])
        mask=np.zeros_like(frame)
        
        cv2.fillPoly(mask,polygons,255)
        masked_image=cv2.bitwise_and(frame,mask)
        return masked_image

    def display_lines(frame,lines):
        line_image=np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1,y1,x2,y2=line.reshape(4)
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        return line_image
    lane_image=np.copy(frame)
    canny=canny(lane_image)
    cropped_image=region_of_interest(canny)

    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=5,maxLineGap=300)
    
    line_image=display_lines(lane_image,lines)
    frame=cv2.addWeighted(lane_image,0.8,line_image,1,1)
    img_item = "2.png"
    cv2.imwrite(img_item, frame)

    cv2.imshow('video', frame)
    
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()