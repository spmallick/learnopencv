import os
import time
import sys
import cv2
import argparse
from frame_differencing import capture_slides_frame_diff
from post_process import remove_duplicates
from utils import resize_image_frame, create_output_directory, convert_slides_to_pdf


# -------------- Initializations ---------------------

FRAME_BUFFER_HISTORY = 15   # Length of the frame buffer history to model background.
DEC_THRESH = 0.75           # Threshold value, above which it is marked foreground, else background.
DIST_THRESH = 100           # Threshold on the squared distance between the pixel and the sample to decide whether a pixel is close to that sample.

MIN_PERCENT = 0.15          # %age threshold to check if there is motion across subsequent frames
MAX_PERCENT = 0.01          # %age threshold to determine if the motion across frames has stopped.

# ----------------------------------------------------


def capture_slides_bg_modeling(video_path, output_dir_path, type_bgsub, history, threshold, MIN_PERCENT_THRESH, MAX_PERCENT_THRESH):

    print(f"Using {type_bgsub} for Background Modeling...")
    print('---'*10)

   
    if type_bgsub == 'GMG':
        bg_sub = cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=history, decisionThreshold=threshold)

    elif type_bgsub == 'KNN':
        bg_sub = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=threshold, detectShadows=False) 
        

    capture_frame = False
    screenshots_count = 0

    # Capture video frames.
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print('Unable to open video file: ', video_path)
        sys.exit()
     
    
    start = time.time()
    # Loop over subsequent frames.
    while cap.isOpened():
        
        ret, frame = cap.read()

        if not ret:
            break

        # Create a copy of the original frame.
        orig_frame = frame.copy() 
        # Resize the frame keeping aspect ratio.
        frame = resize_image_frame(frame, resize_width=640) 

        # Apply each frame through the background subtractor.
        fg_mask = bg_sub.apply(frame) 

        # Compute the percentage of the Foreground mask."
        p_non_zero = (cv2.countNonZero(fg_mask) / (1.0 * fg_mask.size)) * 100

        # %age of non-zero pixels < MAX_PERCENT_THRESH, implies motion has stopped.
        # Therefore, capture the frame.
        if p_non_zero < MAX_PERCENT_THRESH and not capture_frame:
            capture_frame = True

            screenshots_count += 1
            
            png_filename = f"{screenshots_count:03}.png"
            out_file_path = os.path.join(output_dir_path, png_filename)
            print(f"Saving file at: {out_file_path}")
            cv2.imwrite(out_file_path, orig_frame)
            

        # p_non_zero >= MIN_PERCENT_THRESH, indicates motion/animations.
        # Hence wait till the motion across subsequent frames has settled down.
        elif capture_frame and p_non_zero >= MIN_PERCENT_THRESH:
            capture_frame = False


    end_time = time.time()
    print('***'*10,'\n')
    print("Statistics:")
    print('---'*10)
    print(f'Total Time taken: {round(end_time-start, 3)} secs')
    print(f'Total Screenshots captured: {screenshots_count}')
    print('---'*10,'\n')
    
    # Release Video Capture object.
    cap.release()



if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="This script is used to convert video frames into slide PDFs.")
    parser.add_argument("-v", "--video_file_path", help="Path to the video file", type=str)
    parser.add_argument("-o", "--out_dir", default = 'output_results', help="Path to the output directory", type=str)
    parser.add_argument("--type", help = "type of background subtraction to be used", default = 'GMG', 
                        choices=['Frame_Diff', 'GMG', 'KNN'], type=str)
    parser.add_argument("--no_post_process", action="store_true", default=False, help="flag to apply post processing or not")
    parser.add_argument("--convert_to_pdf", action="store_true", default=False, help="flag to convert the entire image set to pdf or not")
    args = parser.parse_args()


    video_path = args.video_file_path
    output_dir_path = args.out_dir
    type_bg_sub = args.type


    output_dir_path = create_output_directory(video_path, output_dir_path, type_bg_sub)

    if type_bg_sub.lower() == 'frame_diff':
        capture_slides_frame_diff(video_path, output_dir_path)
    
    else:

        if type_bg_sub.lower() == 'gmg':
            thresh = DEC_THRESH
        elif type_bg_sub.lower() == 'knn':
            thresh = DIST_THRESH

        capture_slides_bg_modeling(video_path, output_dir_path, type_bgsub=type_bg_sub,
                                   history=FRAME_BUFFER_HISTORY, threshold=thresh,
                                   MIN_PERCENT_THRESH=MIN_PERCENT, MAX_PERCENT_THRESH=MAX_PERCENT)

    # Perform post-processing using difference hashing technique to remove duplicate slides.
    if not args.no_post_process:
        remove_duplicates(output_dir_path)


    if args.convert_to_pdf:
        convert_slides_to_pdf(video_path, output_dir_path)

    
    