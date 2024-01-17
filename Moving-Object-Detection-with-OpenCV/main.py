import cv2

input_video = "sample/car.mp4"


# video Inference


def vid_inf(vid_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(vid_path)
    # get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_video = "output_recorded.mp4"

    # create the `VideoWriter()` object
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

    # Create Background Subtractor MOG2 object
    backSub = cv2.createBackgroundSubtractorMOG2()

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video file")
    count = 0
    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Apply background subtraction
            fg_mask = backSub.apply(frame)

            # apply global threshol to remove shadows
            retval, mask_thresh = cv2.threshold(fg_mask, 180, 255, cv2.THRESH_BINARY)

            # set the kernal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Apply erosion
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)

            # Find contours
            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            min_contour_area = 500  # Define your minimum area threshold
            # filtering contours using list comprehension
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            frame_out = frame.copy()
            for cnt in large_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                frame_out = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 3)

            # saving the video file
            out.write(frame_out)

            # Display the resulting frame
            cv2.imshow("Frame_final", frame_out)

            # Press Q on keyboard to exit
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break
        else:
            break

    # When everything done, release the video capture and writer object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


vid_inf(input_video)
