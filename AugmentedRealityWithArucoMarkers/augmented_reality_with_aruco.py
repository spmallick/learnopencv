# This code is written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:   python3 augmented_reality_with_aruco.py --image=test.jpg
#                  python3 augmented_reality_with_aruco.py --video=test.mp4

import cv2 as cv
#from cv2 import aruco
import argparse
import sys
import os.path
import numpy as np

parser = argparse.ArgumentParser(description='Augmented Reality using Aruco markers in OpenCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
args = parser.parse_args()

im_src = cv.imread("new_scenery.jpg");

outputFile = "ar_out_py.avi"
if (args.image):
    # Open the image file
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_ar_out_py.jpg'
elif (args.video):
    # Open the video file
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_ar_out_py.avi'
    print("Storing it as :", outputFile)
else:
    # Webcam input
    cap = cv.VideoCapture(0)

# Get the video writer initialized to save the output video
if (not args.image):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 28, (round(2*cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

winName = "Augmented Reality using Aruco markers in OpenCV"

while cv.waitKey(1) < 0:
    try:
        # get frame from the video
        hasFrame, frame = cap.read()
        
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("Output file is stored as ", outputFile)
            cv.waitKey(3000)
            break

        #Load the dictionary that was used to generate the markers.
        dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)
        
        # Initialize the detector parameters using default values
        parameters =  cv.aruco.DetectorParameters_create()
        
        # Detect the markers in the image
        markerCorners, markerIds, rejectedCandidates = cv.aruco.detectMarkers(frame, dictionary, parameters=parameters)

        index = np.squeeze(np.where(markerIds==25));
        refPt1 = np.squeeze(markerCorners[index[0]])[1];
        
        index = np.squeeze(np.where(markerIds==33));
        refPt2 = np.squeeze(markerCorners[index[0]])[2];

        distance = np.linalg.norm(refPt1-refPt2);
        
        scalingFac = 0.02;
        pts_dst = [[refPt1[0] - round(scalingFac*distance), refPt1[1] - round(scalingFac*distance)]];
        pts_dst = pts_dst + [[refPt2[0] + round(scalingFac*distance), refPt2[1] - round(scalingFac*distance)]];
        
        index = np.squeeze(np.where(markerIds==30));
        refPt3 = np.squeeze(markerCorners[index[0]])[0];
        pts_dst = pts_dst + [[refPt3[0] + round(scalingFac*distance), refPt3[1] + round(scalingFac*distance)]];

        index = np.squeeze(np.where(markerIds==23));
        refPt4 = np.squeeze(markerCorners[index[0]])[0];
        pts_dst = pts_dst + [[refPt4[0] - round(scalingFac*distance), refPt4[1] + round(scalingFac*distance)]];

        pts_src = [[0,0], [im_src.shape[1], 0], [im_src.shape[1], im_src.shape[0]], [0, im_src.shape[0]]];
        
        pts_src_m = np.asarray(pts_src)
        pts_dst_m = np.asarray(pts_dst)

        # Calculate Homography
        h, status = cv.findHomography(pts_src_m, pts_dst_m)
        
        # Warp source image to destination based on homography
        warped_image = cv.warpPerspective(im_src, h, (frame.shape[1],frame.shape[0]))
        
        # Prepare a mask representing region to copy from the warped image into the original frame.
        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8);
        cv.fillConvexPoly(mask, np.int32([pts_dst_m]), (255, 255, 255), cv.LINE_AA);

        # Erode the mask to not copy the boundary effects from the warping
        element = cv.getStructuringElement(cv.MORPH_RECT, (3,3));
        mask = cv.erode(mask, element, iterations=3);

        # Copy the mask into 3 channels.
        warped_image = warped_image.astype(float)
        mask3 = np.zeros_like(warped_image)
        for i in range(0, 3):
            mask3[:,:,i] = mask/255

        # Copy the warped image into the original frame in the mask region.
        warped_image_masked = cv.multiply(warped_image, mask3)
        frame_masked = cv.multiply(frame.astype(float), 1-mask3)
        im_out = cv.add(warped_image_masked, frame_masked)
        
        # Showing the original image and the new output image side by side
        concatenatedOutput = cv.hconcat([frame.astype(float), im_out]);
        cv.imshow("AR using Aruco markers", concatenatedOutput.astype(np.uint8))
        
        # Write the frame with the detection boxes
        if (args.image):
            cv.imwrite(outputFile, concatenatedOutput.astype(np.uint8));
        else:
            vid_writer.write(concatenatedOutput.astype(np.uint8))


    except Exception as inst:
        print(inst)

cv.destroyAllWindows()
if 'vid_writer' in locals():
    vid_writer.release()
    print('Video writer released..')
