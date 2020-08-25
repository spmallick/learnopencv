import cv2
import numpy as np

#create VideoCapture object and read from video file
cap = cv2.VideoCapture('dataset/cars.mp4')

while True:
    #capture frame by frame
    ret, frame = cap.read()
    #convert video into gray scale of each frames
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
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
    
    #lane detection ends
    
    cv2.imshow('video', frame)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
#release the videocapture object
cap.release()
#close all the frames
cv2.destroyAllWindows()