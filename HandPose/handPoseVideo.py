import cv2
import time
import numpy as np
import scipy
from scipy import signal

protoFile = "hand/pose_deploy.prototxt"
weightsFile = "hand/pose_iter_102000.caffemodel"
nPoints = 22
POSE_PAIRS = [ [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],[0,17],[17,18],[18,19],[19,20] ]

threshold = 0.2

input_source = "asl.mp4"
cap = cv2.VideoCapture(input_source)
hasFrame, frame = cap.read()

frameWidth = frame.shape[1]
frameHeight = frame.shape[0]

aspect_ratio = frameWidth/frameHeight

inHeight = 368
inWidth = int(((aspect_ratio*inHeight)*8)//8)

vid_writer = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (frame.shape[1],frame.shape[0]))

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)
k = 0

smoothed_data, data, previous_points, frame_number = [], [], [], 0
for i in range(nPoints): previous_points.append((0,0))
# Smoothing parameters - choose a window length of about 1/2 to 1/4 the fps (so 60 fps --> window length of 33)
window_length, exponent_value = 33,2 # Window length must be odd

### This is going to take some time before the party get's started! The downside of smoothing is that
### data from the past and the future is required to smooth data in the present. This means that all the frames
### in the video must be processed before smoothing the data and displaying the result. This method is therefore
### not suitable for realtime results.
while 1:
    k+=1
    t = time.time()
    hasFrame, frame = cap.read()
    frameCopy = np.copy(frame)
    if not hasFrame:
        break

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                              (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()

    print("forward = {}".format(time.time() - t))

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]
        probMap = cv2.resize(probMap, (frameWidth, frameHeight))

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        if prob > threshold :
            cv2.circle(frameCopy, (int(point[0]), int(point[1])), 6, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(point[0]), int(point[1])))
        else :
            # Add the last known point (ex: if thumb is not detected, use thumb position from previous frame)
            points.append(previous_points[i])

    # Save the data from the model - data is a list of lists. Each element is a list containing the 22 coordinates for the hand.
    data.append(points)
    
    previous_points = points

    print("total = {}".format(time.time() - t))

# Re-capture the source, so that the video starts at the beginning again
cap = cv2.VideoCapture(input_source)

# Smooth it out
# Split the data so that just the x values for the first point are made into a list
smoothed_data_in_series = []
for point_index in range(nPoints): # Iterate through each point (wrist, thumb, etc...)
    data_point_series_x = []
    data_point_series_y = []
    for values in data: # Iterate through the series of data (each frame of video)
        data_point_series_x.append(values[point_index][0])
        data_point_series_y.append(values[point_index][1])
    # Smooth the x and y values
    smoothed_data_point_series_x = signal.savgol_filter(data_point_series_x, window_length, exponent_value)
    smoothed_data_point_series_y = signal.savgol_filter(data_point_series_y, window_length, exponent_value)
    smoothed_data_in_series.append(smoothed_data_point_series_x)
    smoothed_data_in_series.append(smoothed_data_point_series_y)
    
# Now the data is sepearted into 44 lists (two lists for each of the 22 locations of the hand, time to zip()
for current_frame_number in range(len(smoothed_data_in_series[0])):
    frame_values = []
    for point_index in range(nPoints):
        x = smoothed_data_in_series[point_index*2][current_frame_number]
        y = smoothed_data_in_series[point_index*2+1][current_frame_number]
        frame_values.append((x,y))
    smoothed_data.append(frame_values)

# Iterate through each frame of data
for values in smoothed_data:
    
    hasFrame, img = cap.read()

    # When data is smoothed, floats are introduced, these must be eliminated becase cv2.circle requries integers
    values = np.array(values, int)  
    
    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]
        cv2.line(img, (values[partA][0], values[partA][1]),(values[partB][0], values[partB][1]), (32, 255, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(img, (values[partA][0], values[partA][1]), 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
        cv2.circle(img, (values[partB][0], values[partB][1]), 2, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

    cv2.imshow('Output-Skeleton', img)
    cv2.waitKey(0)
    vid_writer.write(img)
    
cv2.destroyAllWindows()
vid_writer.release()
cap.release()

