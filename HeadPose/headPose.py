import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils import face_utils
import dlib

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
while True:
    # load the input image and convert it to grayscale
    _, image = cap.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    # detect faces in the grayscale image
    rects = detector(gray, 0)
    
    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            
        size=gray.shape
        image_points = np.array([(shape[33][0],shape[33][1]),     # Nose tip
                                 (shape[8][0], shape[8][1]),     # Chin
                                 (shape[36][0], shape[36][1]),     # Left eye left corner
                                 (shape[45][0], shape[45][1]),     # Right eye right corner
                                 (shape[48][0], shape[48][1]),     # Left Mouth corner
                                 (shape[54][0], shape[54][1])      # Right mouth corner
                                ], dtype="double")
        
        # 3D model points.
        model_points = np.array([(0.0, 0.0, 0.0),             # Nose tip
                                 (0.0, -330.0, -65.0),        # Chin
                                 (-225.0, 170.0, -135.0),     # Left eye left corner
                                 (225.0, 170.0, -135.0),      # Right eye right corne
                                 (-150.0, -150.0, -125.0),    # Left Mouth corner
                                 (150.0, -150.0, -125.0)      # Right mouth corner
                                ])
        
        # Camera internals
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                  [0, focal_length, center[1]],
                                  [0, 0, 1]], dtype = "double"
                                )
        
        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        
        print ("Rotation Vector:\n {0}".format(rotation_vector))
        print ("Translation Vector:\n {0}".format(translation_vector))
        
        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        
        p1 = ( int(image_points[0][0]), int(image_points[0][1]))
        p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
        
        cv2.line(image, p1, p2, (255,0,0), 2)
        
    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

cv2.destroyAllWindows()
cap.release()
