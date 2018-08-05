#!/usr/bin/python
#
# Copyright 2018 BIG VISION LLC ALL RIGHTS RESERVED
# 
from __future__ import print_function
import sys
import cv2
from random import randint

trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']


def createTrackerByName(trackingAlgo):
  # Create a tracker based on tracker name
  if trackingAlgo == trackerTypes[0]:
    tracker = cv2.TrackerBoosting_create()
  elif trackingAlgo == trackerTypes[1]: 
    tracker = cv2.TrackerMIL_create()
  elif trackingAlgo == trackerTypes[2]:
    tracker = cv2.TrackerKCF_create()
  elif trackingAlgo == trackerTypes[3]:
    tracker = cv2.TrackerTLD_create()
  elif trackingAlgo == trackerTypes[4]:
    tracker = cv2.TrackerMedianFlow_create()
  elif trackingAlgo == trackerTypes[5]:
    tracker = cv2.TrackerGOTURN_create()
  elif trackingAlgo == trackerTypes[6]:
    tracker = cv2.TrackerMOSSE_create()
  elif trackingAlgo == trackerTypes[7]:
    tracker = cv2.TrackerCSRT_create()
  else:
    tracker = None
    print('Incorrect tracker name')
    print('Available trackers are:')
    for t in trackerTypes:
      print(t)
    
  return tracker

if __name__ == '__main__':

  print("Default tracking algoritm is KCF \n"
        "Available tracking algorithms are:\n")
  for t in trackerTypes:
      print(t)      

  # set default values for tracking algorithm and video
  videoPath = "videos/run.mp4"
  trackingAlgo = "CSRT"
  
  # read videoPath and tracking Algo from arguments
  if len(sys.argv) == 2:
    videoPath = sys.argv[1]
  elif len(sys.argv) == 3:
    videoPath = sys.argv[1]
    trackingAlgo = sys.argv[2]

  outputVideo = "results/multiTracker-" + trackingAlgo + ".avi"

  ## Initialize MultiTracker
  # There are two ways you can initialize multitracker
  # 1. tracker = cv2.MultiTracker("CSRT")
  # All the trackers added to this multitracker
  # will use CSRT algorithm as default
  # 2. tracker = cv2.MultiTracker()
  # No default algorithm specified

  # Initialize MultiTracker with tracking algo
  multiTracker = cv2.MultiTracker_create()
  
  # create a video capture object to read videos
  cap = cv2.VideoCapture(videoPath)
  frame_width = int(cap.get(3))
  frame_height = int(cap.get(4))
  video = cv2.VideoWriter(outputVideo,cv2.VideoWriter_fourcc('M','J','P','G'), 25, (int(frame_width),int(frame_height)))
  
  # read first frame
  success, frame = cap.read()
  # quit if unable to read the video file
  if not success:
    print('Failed to read video')
    sys.exit(1)

  ## Select boxes
  init = False
  bboxes = []
  trackers = []
  colors = [] 


  # OpenCV's selectROI function doesn't work for selecting multiple objects in Python
  # So we will call this function in a loop till we are done selecting all objects
  while True:
    # draw bounding boxes over objects
    # selectROI's default behaviour is to draw box starting from the center
    # when fromCenter is set to false, you can draw box starting from top left corner
    bbox = cv2.selectROI('Tracking', frame)
    bboxes.append(bbox)
    colors.append((randint(64, 255), randint(64, 255), randint(64, 255)))
    # trackers.append(createTrackerByName(trackingAlgo))
    print("Press q to quit selecting boxes and start tracking")
    print("Press any other key to select next object")
    k = cv2.waitKey(0) & 0xFF
    if (k == 113):  # q is pressed
      break
  
  print('Selected bounding boxes {}'.format(bboxes))

  # process video and track objects
  while cap.isOpened():
    success, frame = cap.read()
    if not success:
      break
    
    # intialize and add trackers for first frame
    if not init:
      for bbox in bboxes:
        multiTracker.add(createTrackerByName(trackingAlgo), frame, bbox)
      init = True

    # get updated location of objects in subsequent frames
    success, boxes = multiTracker.update(frame)

    # draw tracked objects
    for i, newbox in enumerate(boxes):
      p1 = (int(newbox[0]), int(newbox[1]))
      p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
      cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    # show frame
    cv2.imshow('Tracking', frame)
    video.write(frame)

    # quit on ESC button
    if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
      break
  video.release()
