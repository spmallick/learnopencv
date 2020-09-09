import argparse
import time

import cv2
import numpy as np


def main(video, device):

    # init dict to track time for every stage at each iteration
    timers = {
        "full pipeline": [],
        "reading": [],
        "pre-process": [],
        "optical flow": [],
        "post-process": [],
    }

    # init video capture with video
    cap = cv2.VideoCapture(video)

    # get default video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    # get total number of video frames
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # read the first frame
    ret, previous_frame = cap.read()

    # proceed if frame reading was successful
    if ret:
        # resize frame
        frame = cv2.resize(previous_frame, (960, 540))
        # convert to gray
        previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # create hsv output for optical flow
        hsv = np.zeros_like(frame)
        # set saturation to a maximum value
        hsv[..., 1] = 255

        while True:
            # start full pipeline timer
            start_full_time = time.time()

            # start reading timer
            start_read_time = time.time()

            # capture frame-by-frame
            ret, current_frame = cap.read()

            # end reading timer
            end_read_time = time.time()
            # add elapsed iteration time
            timers["reading"].append(end_read_time - start_read_time)

            # if frame reading was not successful, break
            if not ret:
                break

            # start pre-process timer
            start_pre_time = time.time()
            # resize frame
            frame = cv2.resize(current_frame, (960, 540))

            # convert to gray
            current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if device == "cpu":
                # end pre-process timer
                end_pre_time = time.time()
                # add elapsed iteration time
                timers["pre-process"].append(end_pre_time - start_pre_time)

                # start optical flow timer
                start_of = time.time()
                # calculate optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    previous_frame, current_frame, None, 0.5, 5, 15, 3, 5, 1.2, 0,
                )
                # end of timer
                end_of = time.time()
                # add elapsed iteration time
                timers["optical flow"].append(end_of - start_of)

            else:
                # move both frames to GPU
                cu_previous = cv2.cuda_GpuMat()
                cu_current = cv2.cuda_GpuMat()

                cu_previous.upload(previous_frame)
                cu_current.upload(current_frame)

                # end pre-process timer
                end_pre_time = time.time()
                # add elapsed iteration time
                timers["pre-process"].append(end_pre_time - start_pre_time)

                # start optical flow timer
                start_of = time.time()
                # create optical flow instance
                flow = cv2.cuda_FarnebackOpticalFlow.create(
                    5, 0.5, False, 15, 3, 5, 1.2, 0,
                )
                # calculate optical flow
                flow = cv2.cuda_FarnebackOpticalFlow.calc(
                    flow, cu_previous, cu_current, None,
                )
                # sent result from GPU back to CPU
                flow = flow.download()

                # end of timer
                end_of = time.time()
                # add elapsed iteration time
                timers["optical flow"].append(end_of - start_of)

            # start post-process timer
            start_post_time = time.time()

            # convert from cartesian to polar coordinates to get magnitude and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            # set hue according to the angle of optical flow
            hsv[..., 0] = ang * 180 / np.pi / 2
            # set value according to the normalized magnitude of optical flow
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            # convert hsv to bgr
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # end post-process timer
            end_post_time = time.time()
            # add elapsed iteration time
            timers["post-process"].append(end_post_time - start_post_time)

            # end full pipeline timer
            end_full_time = time.time()
            # add elapsed iteration time
            timers["full pipeline"].append(end_full_time - start_full_time)

            # visualization
            cv2.imshow("original", frame)
            cv2.imshow("result", bgr)
            k = cv2.waitKey(1)
            if k == 27:
                break

            # update previous_frame value
            previous_frame = current_frame

    # release the capture
    cap.release()
    # destroy all windows
    cv2.destroyAllWindows()

    # print results
    print("Number of frames : ", num_frames)

    # elapsed time at each stage
    print("Elapsed time")
    for stage, seconds in timers.items():
        print("-", stage, ": {:0.3f} seconds".format(sum(seconds)))

    # calculate frames per second
    print("Default video FPS : {:0.3f}".format(fps))

    of_fps = (num_frames - 1) / sum(timers["optical flow"])
    print("Optical flow FPS : {:0.3f}".format(of_fps))

    full_fps = (num_frames - 1) / sum(timers["full pipeline"])
    print("Full pipeline FPS : {:0.3f}".format(full_fps))


if __name__ == "__main__":

    # init argument parser
    parser = argparse.ArgumentParser(description="OpenCV CPU/GPU Comparison")

    parser.add_argument(
        "--video", help="path to .mp4 video file", required=True, type=str,
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "gpu"],
        help="device to inference on",
    )

    # parsing script arguments
    args = parser.parse_args()
    video = args.video
    device = args.device

    # output passed arguments
    print("Configuration")
    print("- device : ", device)
    print("- video file : ", video)

    # run pipeline
    main(video, device)
