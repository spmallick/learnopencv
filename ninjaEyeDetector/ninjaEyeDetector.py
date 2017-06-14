import numpy as np
import cv2

# Load the Cascade Classifier Xml file
face_cascade = cv2.CascadeClassifier("cascade/mallick_haarcascade_frontalface_default.xml")

# Specifying minimum and maximum size parameters
MIN_FACE_SIZE = 100
MAX_FACE_SIZE = 300

#Create a VideoCapture object
cap = cv2.VideoCapture(0)
 
# Check if camera opened successfully
if (cap.isOpened() == False): 
  print("Unable to read camera feed")

while(True):
  # Reading each frame
  ret, frameBig = cap.read()
 
  # If frame opened successfully
  if ret == True: 
     
    # Fixing the scaling factor
    scale = 640.0/frameBig.shape[1]
    
    # Resizing the image
    frame = cv2.resize(frameBig,None,fx = scale,fy = scale, interpolation = cv2.INTER_LINEAR)
  
    # Converting to grayscale
    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(frameGray, scaleFactor=1.1, minNeighbors=5,flags=0, minSize=(MIN_FACE_SIZE,MIN_FACE_SIZE),maxSize=(MAX_FACE_SIZE,MAX_FACE_SIZE))
  
    # Loop over each detected face
    for i in xrange (0,len(faces)):
    
      # Dimension parameters for bounding rectangle for face
      x,y,width,height = faces[i];

      # Calculating the dimension parameters for eyes from the dimensions parameters of the face
      ex,ey,ewidth,eheight = int(x + 0.125*width), int(y + 0.25 * height), int(0.75 * width), int(0.25 * height)             
      
      # Drawing the bounding rectangle around the face
      cv2.rectangle(frame, (ex,ey),(ex+ewidth,ey+eheight),(128,255,0), 2)    

    # Display the resulting frame    
    cv2.imshow('Ninja Eye Detector',frame)
 
    # Press ESC on keyboard to stop tracking
    key = cv2.waitKey(1)
 
    if (key==27):

      break
 
  # Break the loop
  else:
    break 
# Release VideoCapture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows() 
