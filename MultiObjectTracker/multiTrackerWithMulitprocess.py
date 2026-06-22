"""
    Intro: This project targets utilizing power of multicore cpu to speed up the multi object detection based on in-built opencv functions.
    Problem: With each new object tracking task, the speed decreases significantly
    Improvement: Using separate process for each individual tracker and clearing the closed process.
"""
import cv2
from random import randint
import multiprocessing
import argparse

# trackerTypes = ['BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

def new_tracker(label, box, frame, inputQ, outputQ):
    """ method to be executed in multiple process as Service """
    
    # initializing single tracker
    tracker = cv2.TrackerCSRT_create() # creating instance of tracker
    tracker.init(frame, box)

    # now deamon process
    while True: # loop infinitely
        frame = inputQ.get() # attempt to grab new frame
        if frame is not None:
            success, box = tracker.update(frame) # updating the frame and getting next location box

            if success: # got new location successfully
                outputQ.put(box)
            else:
                outputQ.put("failed")
                print("Tracking fialed label: ", label)
                break


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", default="videos/run.mp4", help="path to video file")
    arg = parser.parse_args()
    video_path = arg.video

    # initializing list for containing queues, which will track every object
    inputQ = list()
    outputQ = list()
    activeP = list() # active process list
    jobs = list() # list for handling processes instance

    cap = cv2.VideoCapture(video_path) # creating video reader

    _, first_f = cap.read() # reading first frame for initialization

    bboxes = [] 
    colors = []

    while True: # looping to mark initial detections
        bbox = cv2.selectROI('MultiTracker', first_f)
        bboxes.append(bbox)
        colors.append((randint(0, 255), randint(0, 255), randint(0, 255)))
        print("Press q to quit selecting boxes and start tracking")
        print("Press any other key to select next object")
        k = cv2.waitKey(0) & 0xFF
        if (k == 113):  # q is pressed
            break

    print('Selected bounding boxes {}'.format(bboxes))

    # Initialize Multiple tracking using multiprocessing 
    for i, bbox in enumerate(bboxes):
        # create new input and output queues
        iq = multiprocessing.Queue()
        oq = multiprocessing.Queue()
        inputQ.append(iq)
        outputQ.append(oq)
        activeP.append(i)

        # spawn the new deamon process for tracking
        p = multiprocessing.Process(
            target = new_tracker,
            args = (i, bbox, first_f, iq, oq)
        )

        jobs.append(p)

        p.deamon = True
        p.start()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # start timer
        timer = cv2.getTickCount()

        # passing new frame to all the input queues
        for idx in activeP:
            iq = inputQ[idx]
            iq.put(frame)


        # draw tracked objects
        for idx in activeP:
            oq = outputQ[idx]
            newbox = oq.get()

            if newbox == 'failed':
                activeP.remove(idx)
                inputQ[idx].close()
                outputQ[idx].close()
                # jobs[idx].join()
                continue

            p1 = (int(newbox[0]), int(newbox[1]))
            p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
            cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

        # get FPS
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
        # show frame
        cv2.imshow('MultiTracker', frame)

        # quit on q button
        if cv2.waitKey(10) & 0xFF == ord('q'):  # q pressed
            break

    for p in jobs: # cleaning processes after video is completed
        p.terminate()
        p.join()

    cv2.destroyAllWindows()
