# !python3 -m pip install depthai

import os
import json
import numpy as np
import cv2
from pathlib import Path
import depthai as dai
import time

# Define path to the model and configuration
YOLOV8N_MODEL = "Yolov8-2022.1-blob/yolov8n-pothole-best_openvino_2022.1_8shave.blob"  #Adjust path accordingly
YOLOV8N_CONFIG = "Yolov8-2022.1-blob/yolov8n-pothole-best.json" #Adjust path accordingly

OUTPUT_VIDEO = "vid_result/960-oak-d-live_video.mp4" #Adjust path accordingly

CAMERA_PREV_DIM = (960, 960)
LABELS = ["Pot-hole"]

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)
    
def create_camera_pipeline(config_path, model_path):
    pipeline = dai.Pipeline()
    model_config = load_config(config_path)
    nnConfig = model_config.get("nn_config", {})
    metadata = nnConfig.get("NN_specific_metadata", {})
    classes = metadata.get("classes", {})
    coordinates = metadata.get("coordinates", {})
    anchors = metadata.get("anchors", {})
    anchorMasks = metadata.get("anchor_masks", {})
    iouThreshold = metadata.get("iou_threshold", {})
    confidenceThreshold = metadata.get("confidence_threshold", {})

    # Create camera node
    camRgb = pipeline.create(dai.node.ColorCamera)
    camRgb.setPreviewSize(CAMERA_PREV_DIM[0], CAMERA_PREV_DIM[1])
    camRgb.setInterleaved(False)
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)

    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)

    nnOut.setStreamName("nn")

    detectionNetwork.setConfidenceThreshold(confidenceThreshold)
    detectionNetwork.setNumClasses(classes)
    detectionNetwork.setCoordinateSize(coordinates)
    detectionNetwork.setAnchors(anchors)
    detectionNetwork.setAnchorMasks(anchorMasks)
    detectionNetwork.setIouThreshold(iouThreshold)
    detectionNetwork.setBlobPath(model_path)
    detectionNetwork.setNumInferenceThreads(2)
    detectionNetwork.input.setBlocking(False)

    # Linking
    camRgb.preview.link(detectionNetwork.input)
    detectionNetwork.out.link(nnOut.input)

    return pipeline

def annotate_frame(frame, detections, fps):
    color = (0, 0, 255)
    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    
    # Annotate the frame with the FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
    resized = cv2.resize(arr, shape)
    return resized.transpose(2, 0, 1)

def frame_norm(frame, bbox):
    norm_vals = np.full(len(bbox), frame.shape[0])
    norm_vals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

# Create pipeline
pipeline = create_camera_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Define the queue that will be used to receive the neural network output
    detectionNN = device.getOutputQueue("nn", maxSize=4, blocking=False)

    # Video writer to save the output video
    fps = 30  # Assuming 30 FPS for the OAK-D camera
    frame_width, frame_height = CAMERA_PREV_DIM
    out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    start_time = time.time()
    frame_count = 0

    while True:
        inDet = detectionNN.get()
        detections = []
        if inDet is not None:
            detections = inDet.detections
            print("Detections", detections)
        
        # Retrieve the frame from the camera preview
        frame = inDet.getFrame()
        frame_count += 1

        # Calculate the FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Annotate the frame with detections and FPS
        frame = annotate_frame(frame, detections, fps)

        # Display the frame
        cv2.imshow("Frame", frame)

        # Write the frame to the output video
        out.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

out.release()
cv2.destroyAllWindows()

print(f"[INFO] Processed live stream and saved to {OUTPUT_VIDEO}")
