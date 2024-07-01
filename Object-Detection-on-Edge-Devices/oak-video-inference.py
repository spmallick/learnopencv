#Install Dependencies
# !python3 -m pip install depthai

import os
import json
import numpy as np
import cv2
from pathlib import Path
import depthai as dai
import time

# Define path to the model, test data directory, and results

YOLOV8N_MODEL = "/home/jaykumaran/Blogs/Poth-hole-Detection/Final Media/Yolov8-2022.1-blob/yolov8n-pothole-best_openvino_2022.1_8shave.blob" #Adjust path accordingly
YOLOV8N_CONFIG = "/home/jaykumaran/Blogs/Poth-hole-Detection/Final Media/Yolov8-2022.1-blob/yolov8n-pothole-best.json" #Adjust path accordingly

INPUT_VIDEO = "videoplayback.mp4"
OUTPUT_VIDEO = "vid_result/960-oak-d-videoplayback_video.mp4"

CAMERA_PREV_DIM = (960, 960)
LABELS = ["Pot-hole"]

def load_config(config_path):
    with open(config_path) as f:
        return json.load(f)
    
    
   
def create_image_pipeline(config_path, model_path):
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

    detectionIN = pipeline.create(dai.node.XLinkIn)
    detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
    nnOut = pipeline.create(dai.node.XLinkOut)

    nnOut.setStreamName("nn")
    detectionIN.setStreamName("detection_in")

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
    detectionIN.out.link(detectionNetwork.input)
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
pipeline = create_image_pipeline(YOLOV8N_CONFIG, YOLOV8N_MODEL)

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_VIDEO), exist_ok=True)

cap = cv2.VideoCapture(INPUT_VIDEO)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Define the queues that will be used in order to communicate with depthai
    detectionIN = device.getInputQueue("detection_in")
    detectionNN = device.getOutputQueue("nn")

    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        image_res = cv2.resize(frame, CAMERA_PREV_DIM)
    
        # Initialize depthai NNData() class which is fed with the image data resized and transposed to model input shape
        nn_data = dai.NNData()
        nn_data.setLayer("input", to_planar(frame, CAMERA_PREV_DIM))
        
        # Send the image to detectionIN queue further passed to the detection network for inference as defined in pipeline
        detectionIN.send(nn_data)
        
        # Fetch the neural network output
        inDet = detectionNN.get()
        detections = []
        if inDet is not None:
            detections = inDet.detections
            print("Detections", detections)

        # Calculate the FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        # Annotate the frame with detections and FPS
        frame = annotate_frame(frame, detections, fps)

        out.write(frame)

cap.release()
out.release()

print(f"[INFO] Processed video {INPUT_VIDEO} and saved to {OUTPUT_VIDEO}")
