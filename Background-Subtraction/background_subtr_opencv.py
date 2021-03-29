import argparse

import cv2


def get_opencv_result(video_to_process):
    # create VideoCapture object for further video processing
    captured_video = cv2.VideoCapture(video_to_process)
    # check video capture status
    if not captured_video.isOpened:
        print("Unable to open: " + video_to_process)
        exit(0)

    # instantiate background subtraction
    background_subtr_method = cv2.bgsegm.createBackgroundSubtractorGSOC()

    while True:
        # read video frames
        retval, frame = captured_video.read()

        # check whether the frames have been grabbed
        if not retval:
            break

        # resize video frames
        frame = cv2.resize(frame, (640, 360))

        # pass the frame to the background subtractor
        foreground_mask = background_subtr_method.apply(frame)
        # obtain the background without foreground mask
        background_img = background_subtr_method.getBackgroundImage()

        # show the current frame, foreground mask, subtracted result
        cv2.imshow("Initial Frames", frame)
        cv2.imshow("Foreground Masks", foreground_mask)
        cv2.imshow("Subtraction Result", background_img)

        keyboard = cv2.waitKey(10)
        if keyboard == 27:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_video",
        type=str,
        help="Define the full input video path",
        default="space_traffic.mp4",
    )

    # parse script arguments
    args = parser.parse_args()

    # start BS-pipeline
    get_opencv_result(args.input_video)
