import cv2
import ntpath
from .utils import filter_persons, draw_keypoints
from .lstm import WINDOW_SIZE
import time

import numpy as np
import torch
import torch.nn.functional as F

LABELS = {
    0: "JUMPING",
    1: "JUMPING_JACKS",
    2: "BOXING",
    3: "WAVING_2HANDS",
    4: "WAVING_1HAND",
    5: "CLAPPING_HANDS"
}

# how many frames to skip while inferencing
# configuring a higher value will result in better FPS (frames per rate), but accuracy might get impacted
SKIP_FRAME_COUNT = 0

# analyse the video
def analyse_video(pose_detector, lstm_classifier, video_path):
    # open the video
    cap = cv2.VideoCapture(video_path)
    # width of image frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height of image frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames per second of the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # total number of frames in the video
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # video output codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # extract the file name from video path
    file_name = ntpath.basename(video_path)
    # video writer
    vid_writer = cv2.VideoWriter('res_{}'.format(
        file_name), fourcc, 30, (width, height))
    # counter
    counter = 0
    # buffer to keep the output of detectron2 pose estimation
    buffer_window = []
    # start time
    start = time.time()
    label = None
    # iterate through the video
    while True:
        # read the frame
        ret, frame = cap.read()
        # return if end of the video
        if ret == False:
            break
        # make a copy of the frame
        img = frame.copy()
        if(counter % (SKIP_FRAME_COUNT+1) == 0):
            # predict pose estimation on the frame
            outputs = pose_detector(frame)
            # filter the outputs with a good confidence score
            persons, pIndicies = filter_persons(outputs)
            if len(persons) >= 1:
                # pick only pose estimation results of the first person.
                # actually, we expect only one person to be present in the video.
                p = persons[0]
                # draw the body joints on the person body
                draw_keypoints(p, img)
                # input feature array for lstm
                features = []
                # add pose estimate results to the feature array
                for i, row in enumerate(p):
                    features.append(row[0])
                    features.append(row[1])

                # append the feature array into the buffer
                # not that max buffer size is 32 and buffer_window operates in a sliding window fashion
                if len(buffer_window) < WINDOW_SIZE:
                    buffer_window.append(features)
                else:
                    # convert input to tensor
                    model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                    # add extra dimension
                    model_input = torch.unsqueeze(model_input, dim=0)
                    # predict the action class using lstm
                    y_pred = lstm_classifier(model_input)
                    prob = F.softmax(y_pred, dim=1)
                    # get the index of the max probability
                    pred_index = prob.data.max(dim=1)[1]
                    # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                    buffer_window.pop(0)
                    buffer_window.append(features)
                    label = LABELS[pred_index.numpy()[0]]
                    #print("Label detected ", label)

        # add predicted label into the frame
        if label is not None:
            cv2.putText(img, 'Action: {}'.format(label),
                        (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        # increment counter
        counter += 1
        # write the frame into the result video
        vid_writer.write(img)
        # compute the completion percentage
        percentage = int(counter*100/tot_frames)
        # return the completion percentage
        yield "data:" + str(percentage) + "\n\n"
    analyze_done = time.time()
    print("Video processing finished in ", analyze_done - start)


def stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("fps ", fps)
    print("width height", width, height)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("tot_frames", tot_frames)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        out_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +
                  out_frame + b'\r\n')
        yield result
    print("finished video streaming")
