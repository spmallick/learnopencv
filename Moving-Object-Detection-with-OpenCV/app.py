import cv2
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# input_video = 'car.mp4'

# video Inference


def vid_inf(vid_path):
    # Create a VideoCapture object
    cap = cv2.VideoCapture(vid_path)

    # get the video frames' width and height for proper saving of videos
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        # print(frame.shape)
        if ret:
            # Apply background subtraction
            fg_mask = backSub.apply(frame)
            # print(fg_mask.shape)
            # cv2.imshow('Frame_bg', fg_mask)

            # apply global threshol to remove shadows
            retval, mask_thresh = cv2.threshold(
                fg_mask, 180, 255, cv2.THRESH_BINARY)
            # cv2.imshow('frame_thresh', mask_thresh)

            # set the kernal
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            # Apply erosion
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
            # cv2.imshow('frame_erode', mask_eroded)

            # Find contours
            contours, hierarchy = cv2.findContours(
                mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)

            min_contour_area = 2000  # Define your minimum area threshold
            large_contours = [
                cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            # frame_ct = cv2.drawContours(frame, large_contours, -1, (0, 255, 0), 2)
            frame_out = frame.copy()
            for cnt in large_contours:
                # print(cnt.shape)
                x, y, w, h = cv2.boundingRect(cnt)
                frame_out = cv2.rectangle(
                    frame, (x, y), (x+w, y+h), (0, 0, 200), 3)
            frame_out_display = cv2.cvtColor(frame_out, cv2.COLOR_BGR2RGB)
            vid = out.write(frame_out)

            # Display the resulting frame
            # cv2.imshow('Frame_final', frame_out)

            # update the count every frame and display every 12th frame
            if not count % 12:
                yield frame_out_display, None
            count += 1

            # Press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # When everything done, release the video capture and writer object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()
    yield None, output_video

# vid_inf(input_video)


# gradio interface
input_video = gr.Video(label="Input Video")
output_frames = gr.Image(label="Output Frames")
output_video_file = gr.Video(label="Output video")
# sample_video=r'sample/car.mp4'

app = gr.Interface(
    fn=vid_inf,
    inputs=[input_video],
    outputs=[output_frames, output_video_file],
    title=f"MotionScope",
    description=f'A gradio app for dynamic video analysis tool that leverages advanced background subtraction and contour detection techniques to identify and track moving objects in real-time.',
    allow_flagging="never",
    examples=[["sample/car.mp4"]],
)
app.queue().launch()