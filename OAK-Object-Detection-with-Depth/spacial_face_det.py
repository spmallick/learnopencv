# Face Detection

# Import required modules
import cv2
import depthai as dai
import time
import blobconverter

# Define Frame
FRAME_SIZE = (640, 400)

# Define NN model name and input size
# If you define the blob make make sure the MODEL_NAME and ZOO_TYPE are None
# DET_INPUT_SIZE = (672, 384)
# model_name = None
# zoo_type = None
# blob_path = "models/face-detection-adas-0001.blob"

DET_INPUT_SIZE = (300, 300)
model_name = "face-detection-retail-0004"
zoo_type = "depthai"
blob_path = None

# DET_INPUT_SIZE = (672, 384)
# model_name = "face-detection-adas-0001"
# zoo_type = "intel"
# blob_path = None


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - RGB camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1])
cam.setInterleaved(False)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

# Define mono camera sources for stereo depth
mono_left = pipeline.createMonoCamera()
mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
mono_right = pipeline.createMonoCamera()
mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create stereo depth node
stereo = pipeline.createStereoDepth()
stereo.setLeftRightCheck(True)

# Linking
mono_left.out.link(stereo.left)
mono_right.out.link(stereo.right)

cam.setBoardSocket(dai.CameraBoardSocket.RGB)

# Convert model from OMZ to blob
if model_name is not None:
    blob_path = blobconverter.from_zoo(
        name=model_name,
        shaves=6,
        zoo_type=zoo_type
    )

# Define face detection NN node
face_spac_det_nn = pipeline.createMobileNetSpatialDetectionNetwork()
face_spac_det_nn.setConfidenceThreshold(0.75)
face_spac_det_nn.setBlobPath(blob_path)
face_spac_det_nn.setDepthLowerThreshold(100)
face_spac_det_nn.setDepthUpperThreshold(5000)

# Define face detection input config
face_det_manip = pipeline.createImageManip()
face_det_manip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
face_det_manip.initialConfig.setKeepAspectRatio(False)

# Linking
cam.preview.link(face_det_manip.inputImage)
face_det_manip.out.link(face_spac_det_nn.input)
stereo.depth.link(face_spac_det_nn.inputDepth)

# Create preview output
x_preview_out = pipeline.createXLinkOut()
x_preview_out.setStreamName("preview")
cam.preview.link(x_preview_out.input)

# Create detection output
det_out = pipeline.createXLinkOut()
det_out.setStreamName('det_out')
face_spac_det_nn.out.link(det_out.input)

# Create preview output
disparity_out = pipeline.createXLinkOut()
disparity_out.setStreamName("disparity")
stereo.disparity.link(disparity_out.input)


def display_info(frame, disp_frame, bbox, coordinates, status, status_color, fps):
    # Display bounding box
    cv2.rectangle(frame, bbox, status_color[status], 2)

    # Create background for showing details
    cv2.rectangle(frame, (5, 5, 175, 100), (50, 0, 0), -1)

    # Display authentication status on the frame
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color[status])

    # Display instructions on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))

    # Display bbox and depth value on the disparity frame
    if coordinates is not None:
        cv2.rectangle(disp_frame, bbox, status_color[status], 2)
        cv2.rectangle(disp_frame, (5, 5, 185, 50), (50, 0, 0), -1)
        _, _, coord_z = coordinates
        cv2.putText(disp_frame, f'Depth: {coord_z}mm', (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))


# Frame count
frame_count = 0

# Placeholder fps value
fps = 0

# Used to record the time when we processed last frames
prev_frame_time = 0

# Used to record the time at which we processed current frames
new_frame_time = 0

# Set status colors
status_color = {
    'Face Detected': (0, 255, 0),
    'No Face Detected': (0, 0, 255)
}

# Start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the right camera frames from the outputs defined above
    q_cam = device.getOutputQueue(name="preview", maxSize=1, blocking=False)

    # Output queue will be used to get nn data from the video frames.
    q_det = device.getOutputQueue(name="det_out", maxSize=1, blocking=False)

    # Output queue will be used to get disparity map from stereo node.
    q_disp = device.getOutputQueue(name="disparity", maxSize=1, blocking=False)

    # # Output queue will be used to get nn data from the video frames.
    # q_bbox_depth_mapping = device.getOutputQueue(name="bbox_depth_mapping_out", maxSize=4, blocking=False)

    while True:
        # Get right camera frame
        in_cam = q_cam.get()
        frame = in_cam.getCvFrame()

        # Get disparity frame
        in_disp = q_disp.get()
        disp_frame = in_disp.getCvFrame()

        # Calculate a multiplier for color mapping disparity map
        disparityMultiplier = 255 / stereo.getMaxDisparity()

        # Colormap disparity for display.
        disp_frame = (disp_frame * disparityMultiplier).astype('uint8')

        # Apply color map to disparity map
        disp_frame = cv2.applyColorMap(disp_frame, cv2.COLORMAP_JET)

        bbox = None
        coordinates = None

        inDet = q_det.tryGet()

        if inDet is not None:
            detections = inDet.detections

            # if face detected
            if len(detections) is not 0:
                detection = detections[0]

                # Correct bounding box
                xmin = max(0, detection.xmin)
                ymin = max(0, detection.ymin)
                xmax = min(detection.xmax, 1)
                ymax = min(detection.ymax, 1)

                # Calculate coordinates
                x = int(xmin*FRAME_SIZE[0])
                y = int(ymin*FRAME_SIZE[1])
                w = int(xmax*FRAME_SIZE[0]-xmin*FRAME_SIZE[0])
                h = int(ymax*FRAME_SIZE[1]-ymin*FRAME_SIZE[1])

                bbox = (x, y, w, h)

                # Get spacial coordinates
                coord_x = detection.spatialCoordinates.x
                coord_y = detection.spatialCoordinates.y
                coord_z = detection.spatialCoordinates.z

                coordinates = (coord_x, coord_y, coord_z)

        # Check if a face was detected in the frame
        if bbox:
            # Face detected
            status = 'Face Detected'
        else:
            # No face detected
            status = 'No Face Detected'

        # Display info on frame
        display_info(frame, disp_frame, bbox, coordinates, status, status_color, fps)

        # Calculate average fps
        if frame_count % 10 == 0:
            # Time when we finish processing last 100 frames
            new_frame_time = time.time()

            # Fps will be number of frame processed in one second
            fps = 1 / ((new_frame_time - prev_frame_time)/10)
            prev_frame_time = new_frame_time

        # Capture the key pressed
        key_pressed = cv2.waitKey(1) & 0xff

        # Stop the program if Esc key was pressed
        if key_pressed == 27:
            break

        # Display the final frame
        cv2.imshow("Face Cam", frame)

        # Display the disparity frame
        cv2.imshow("Disparity Map", disp_frame)

        # Increment frame count
        frame_count += 1

cv2.destroyAllWindows()
