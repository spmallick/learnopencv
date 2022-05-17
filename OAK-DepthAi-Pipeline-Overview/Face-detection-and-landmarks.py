# Face Detection

# Import required modules
import cv2
import depthai as dai
import time
import blobconverter

# Define Frame size
FRAME_SIZE = (640, 400)

# Define Detection NN model name and input size
# If you define the blob make sure the FACE_MODEL_NAME and ZOO_TYPE are None
# DET_INPUT_SIZE = (672, 384)
# FACE_MODEL_NAME = None
# ZOO_TYPE = None
# blob_path = "models/face-detection-adas-0001.blob"
DET_INPUT_SIZE = (300, 300)
FACE_MODEL_NAME = "face-detection-retail-0004"
ZOO_TYPE = "depthai"
blob_path = None

# Define Landmark NN model name and input size
# If you define the blob make sure the LANDMARKS_MODEL_NAME and ZOO_TYPE are None
LANDMARKS_INPUT_SIZE = (48, 48)
LANDMARKS_MODEL_NAME = "landmarks-regression-retail-0009"
LANDMARKS_ZOO_TYPE = "intel"
landmarks_blob_path = None


# Start defining a pipeline
pipeline = dai.Pipeline()

# Define a source - RGB camera
cam = pipeline.createColorCamera()
cam.setPreviewSize(FRAME_SIZE[0], FRAME_SIZE[1])
cam.setInterleaved(False)
cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)

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

# Convert model from OMZ to blob
if LANDMARKS_MODEL_NAME is not None:
    landmarks_blob_path = blobconverter.from_zoo(
        name=LANDMARKS_MODEL_NAME,
        shaves=6,
        zoo_type=LANDMARKS_ZOO_TYPE
    )

# Define landmarks detection NN node
landmarksDetNn = pipeline.createNeuralNetwork()
landmarksDetNn.setBlobPath(landmarks_blob_path)

# Define face detection input config
faceDetManip = pipeline.createImageManip()
faceDetManip.initialConfig.setResize(DET_INPUT_SIZE[0], DET_INPUT_SIZE[1])
faceDetManip.initialConfig.setKeepAspectRatio(False)

# Define landmark detection input config
lndmrksDetManip = pipeline.createImageManip()

# Linking
cam.preview.link(faceDetManip.inputImage)
faceDetManip.out.link(faceDetNn.input)

# Define Script node
# Script node will take the output from the face detection NN as an input and set ImageManipConfig for landmark NN
script = pipeline.create(dai.node.Script)
script.setProcessor(dai.ProcessorType.LEON_CSS)
script.setScriptPath("script.py")

# Linking to Script inputs
cam.preview.link(script.inputs['frame'])
faceDetNn.out.link(script.inputs['face_det_in'])

# Linking Script outputs to landmark ImageManipconfig
script.outputs['manip_cfg'].link(lndmrksDetManip.inputConfig)
script.outputs['manip_img'].link(lndmrksDetManip.inputImage)

# Linking
lndmrksDetManip.out.link(landmarksDetNn.input)

# Create preview output
xOutPreview = pipeline.createXLinkOut()
xOutPreview.setStreamName("preview")
cam.preview.link(xOutPreview.input)

# Create face detection output
xOutDet = pipeline.createXLinkOut()
xOutDet.setStreamName('det_out')
faceDetNn.out.link(xOutDet.input)

# Create cropped face output
xOutCropped = pipeline.createXLinkOut()
xOutCropped.setStreamName('face_out')
lndmrksDetManip.out.link(xOutCropped.input)

# Create landmarks detection output
xOutLndmrks = pipeline.createXLinkOut()
xOutLndmrks.setStreamName('lndmrks_out')
landmarksDetNn.out.link(xOutLndmrks.input)


# Display info on the frame
def display_info(frame, bbox, landmarks, status, status_color, fps):
    # Display bounding box
    cv2.rectangle(frame, bbox, status_color[status], 2)

    # Display landmarks
    if landmarks is not None:
        for landmark in landmarks:
            cv2.circle(frame, landmark, 2, (0, 255, 255), -1)

    # Create background for showing details
    cv2.rectangle(frame, (5, 5, 175, 100), (50, 0, 0), -1)

    # Display authentication status on the frame
    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color[status])

    # Display instructions on the frame
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255))


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
    qCam = device.getOutputQueue(name="preview", maxSize=1, blocking=False)

    # Output queue will be used to get nn detection data from the video frames.
    qDet = device.getOutputQueue(name="det_out", maxSize=1, blocking=False)

    # Output queue will be used to get cropped face region.
    qFace = device.getOutputQueue(name="face_out", maxSize=1, blocking=False)

    # Output queue will be used to get landmarks from the face region.
    qLndmrks = device.getOutputQueue(name="lndmrks_out", maxSize=1, blocking=False)

    while True:
        # Get camera frame
        inCam = qCam.get()
        frame = inCam.getCvFrame()

        bbox = None

        inDet = qDet.tryGet()

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

        # Show cropped face region
        inFace = qFace.tryGet()
        if inFace is not None:
            face = inFace.getCvFrame()
            cv2.imshow("face", face)

        landmarks = None

        # Get landmarks NN output
        inLndmrks = qLndmrks.tryGet()
        if inLndmrks is not None:
            # Get NN layer names
            # print(f"Layer names: {inLndmrks.getAllLayerNames()}")

            # Retrieve landmarks from NN output layer
            landmarks = inLndmrks.getLayerFp16("95")

            x_landmarks = []
            y_landmarks = []

            # Landmarks in following format [x1,y1,x2,y2,..]
            # Extract all x coordinates [x1,x2,..]
            for x_point in landmarks[::2]:
                # Get x coordinate on original frame
                x_point = int((x_point * w) + x)
                x_landmarks.append(x_point)

            # Extract all y coordinates [y1,y2,..]
            for y_point in landmarks[1::2]:
                # Get y coordinate on original frame
                y_point = int((y_point * h) + y)
                y_landmarks.append(y_point)

            # Zip x & y coordinates to get a list of points [(x1,y1),(x2,y2),..]
            landmarks = list(zip(x_landmarks, y_landmarks))

        # Check if a face was detected in the frame
        if bbox:
            # Face detected
            status = 'Face Detected'
        else:
            # No face detected
            status = 'No Face Detected'

        # Display info on frame
        display_info(frame, bbox, landmarks, status, status_color, fps)

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

        # Increment frame count
        frame_count += 1

# Close all output windows
cv2.destroyAllWindows()
