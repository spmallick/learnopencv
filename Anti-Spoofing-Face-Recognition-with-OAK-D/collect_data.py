# Collect dataset

# Import required modules
import cv2
import numpy as np
import depthai as dai
import os
import blobconverter

# Define directory paths
real_face_dir = os.path.join("dataset", "real")
os.makedirs(real_face_dir, exist_ok=True)
spoofed_face_dir = os.path.join("dataset", "spoofed")
os.makedirs(spoofed_face_dir, exist_ok=True)

# Define Detection NN model name and input size
DET_INPUT_SIZE = (300, 300)
FACE_MODEL_NAME = "face-detection-retail-0004"
ZOO_TYPE = "depthai"

# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - two mono (grayscale) cameras
left = pipeline.createMonoCamera()
left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
left.setBoardSocket(dai.CameraBoardSocket.LEFT)

right = pipeline.createMonoCamera()
right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth = pipeline.createStereoDepth()
depth.setConfidenceThreshold(200)
depth.setOutputRectified(True)  # The rectified streams are horizontally mirrored by default
depth.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
depth.setExtendedDisparity(True)  # For better close range depth perception

# Convert model from OMZ to blob
if FACE_MODEL_NAME is not None:
    blob_path = blobconverter.from_zoo(
        name=FACE_MODEL_NAME,
        shaves=6,
        zoo_type=ZOO_TYPE
    )

# Define face detection NN node
faceDetNn = pipeline.createMobileNetDetectionNetwork()
faceDetNn.setConfidenceThreshold(0.75)
faceDetNn.setBlobPath(blob_path)

# Define face detection input config
faceDetManip = pipeline.createImageManip()
faceDetManip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
faceDetManip.initialConfig.setKeepAspectRatio(False)
faceDetManip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)

# Linking
depth.rectifiedRight.link(faceDetManip.inputImage)
faceDetManip.out.link(faceDetNn.input)

# Create face detection output
xOutDet = pipeline.createXLinkOut()
xOutDet.setStreamName('det_out')
faceDetNn.out.link(xOutDet.input)

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
median = dai.StereoDepthProperties.MedianFilter.KERNEL_7x7  # For depth filtering
depth.setMedianFilter(median)

left.out.link(depth.left)
right.out.link(depth.right)

# Create left output
xout_right = pipeline.createXLinkOut()
xout_right.setStreamName("right")
depth.rectifiedRight.link(xout_right.input)

# Create depth output
xout = pipeline.createXLinkOut()
xout.setStreamName("disparity")
depth.disparity.link(xout.input)


# Initialize wlsFilter
# wlsFilter = cv2.ximgproc.createDisparityWLSFilterGeneric(False)
# wlsFilter.setLambda(8000)
# wlsFilter.setSigmaColor(1.5)


# Frame count
count = 0

# Set the number of frames to skip
SKIP_FRAMES = 10

real_count = 0
spoofed_count = 0

save_real = False
save_spoofed = False

# Pipeline defined, now the device is connected to
with dai.Device(pipeline) as device:
    # Start pipeline
    device.startPipeline()

    # Output queue will be used to get the right camera frames from the outputs defined above
    q_right = device.getOutputQueue(name="right", maxSize=4, blocking=False)

    # Output queue will be used to get the disparity frames from the outputs defined above
    q_depth = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    # Output queue will be used to get nn detection data from the video frames.
    qDet = device.getOutputQueue(name="det_out", maxSize=1, blocking=False)

    while True:
        # Get right camera frame
        in_right = q_right.get()
        r_frame = in_right.getFrame()
        # r_frame = cv2.flip(r_frame, flipCode=1)
        # cv2.imshow("right", r_frame)

        # Get depth frame
        in_depth = q_depth.get()  # blocking call, will wait until a new data has arrived
        depth_frame = in_depth.getFrame()
        depth_frame = np.ascontiguousarray(depth_frame)
        depth_frame = cv2.bitwise_not(depth_frame)
        depth_frame = cv2.flip(depth_frame, flipCode=1)

        # Apply wls filter
        # cv2.imshow("without wls filter", cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET))
        # depth_frame = wlsFilter.filter(depth_frame, r_frame)

        # frame is transformed, the color map will be applied to highlight the depth info
        depth_frame_cmap = cv2.applyColorMap(depth_frame, cv2.COLORMAP_JET)
        # frame is ready to be shown
        cv2.imshow("disparity", depth_frame_cmap)

        # Retrieve 'bgr' (opencv format) frame from gray scale
        frame = cv2.cvtColor(r_frame, cv2.COLOR_GRAY2RGB)
        img_h, img_w = frame.shape[0:2]

        bbox = None

        inDet = qDet.tryGet()
        if inDet is not None:
            detections = inDet.detections
            # for detection in detections:
            if len(detections) is not 0:
                detection = detections[0]
                # print(detection.confidence)
                x = int(detection.xmin * img_w)
                y = int(detection.ymin * img_h)
                w = int(detection.xmax * img_w - detection.xmin * img_w)
                h = int(detection.ymax * img_h - detection.ymin * img_h)
                bbox = (x, y, w, h)

        # Set default status
        status_color = (0, 0, 255)

        # Check if a face was detected in the frame
        if bbox:
            # Get face roi from right and depth frames
            face_d = depth_frame[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]]
            face_r = r_frame[max(0, bbox[1]):bbox[1] + bbox[3], max(0, bbox[0]):bbox[0] + bbox[2]]
            cv2.imshow("face_roi", face_d)

        # Display bounding box
        cv2.rectangle(frame, bbox, status_color, 2)

        # Create background for showing details
        cv2.rectangle(frame, (5, 5, 225, 100), (50, 0, 0), -1)

        # Display instructions on the frame
        cv2.putText(frame, 'Press F to save real Face.', (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(frame, 'Press S to save spoofed Face.', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        cv2.putText(frame, 'Press Q to Quit.', (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

        # Capture the key pressed
        key_pressed = cv2.waitKey(1) & 0xff
        # Save face depth map is f was pressed
        if key_pressed == ord('f'):
            save_real = not save_real
            save_spoofed = False
        # Save face depth map is s was pressed
        elif key_pressed == ord('s'):
            save_spoofed = not save_spoofed
            save_real = False
        # Stop the program if q was pressed
        elif key_pressed == ord('q'):
            break

        if bbox:
            if face_d is not None and save_real:
                real_count += 1
                filename = f"real_{real_count}.jpg"
                cv2.imwrite(os.path.join(real_face_dir, filename), face_d)
                # Display authentication status on the frame
                cv2.putText(frame, f"Saving real face: {real_count}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            elif face_d is not None and save_spoofed:
                spoofed_count += 1
                filename = f"spoofed_{spoofed_count}.jpg"
                cv2.imwrite(os.path.join(spoofed_face_dir, filename), face_d)
                # Display authentication status on the frame
                cv2.putText(frame, f"Saving spoofed face: {spoofed_count}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
            else:
                cv2.putText(frame, "Face not saved", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

        # Display the final frame
        cv2.imshow("Data collection Cam", frame)

        count += 1
cv2.destroyAllWindows()
