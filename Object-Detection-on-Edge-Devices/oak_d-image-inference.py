import os
import json
import numpy as np
import cv2
from pathlib import Path
import depthai as dai

# Define path to the model, test data directory, and results
YOLOV8N_MODEL = "yolov8-960-blob-result/best_openvino_2022.1_8shave.blob" #Adjust path accordingly
YOLOV8N_CONFIG = "yolov8-960-blob-result/best.json" #Adjust path accordingly
TEST_DATA = "img-40_jpg.rf.80b05d1760d1169f29d5fd2cf3120ae9.jpg" #Adjust path accordingly
OUTPUT_IMAGES_YOLOv8n = "result/gesture_pred_images_v8n" #Adjust path accordingly
CAMERA_PREV_DIM = (960, 960)
LABELS = ["Pot-hole"]
# LABELS = 

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

    detectionIN.out.link(detectionNetwork.input)
    detectionNetwork.out.link(nnOut.input)

    return pipeline

def annotate_frame(frame, detections):
    color = (0, 0, 255)
    for detection in detections:
        bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
        cv2.putText(frame, LABELS[detection.label], (bbox[0] + 10, bbox[1] + 25), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, color)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
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
os.makedirs(OUTPUT_IMAGES_YOLOv8n, exist_ok=True)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:
    # Define the queues that will be used in order to communicate with depthai
    detectionIN = device.getInputQueue("detection_in")
    detectionNN = device.getOutputQueue("nn")

    # Load the input image and then resize it
    image = cv2.imread(TEST_DATA)
    if image is None:
        raise FileNotFoundError(f"[ERROR] Could not load image {TEST_DATA}")

    image_res = cv2.resize(image, CAMERA_PREV_DIM)
    
    # Initialize depthai NNData() class which is fed with the image data resized and transposed to model input shape
    nn_data = dai.NNData()
    nn_data.setLayer("input", to_planar(image_res, CAMERA_PREV_DIM))
    
    # Send the image to detectionIN queue further passed to the detection network for inference as defined in pipeline
    detectionIN.send(nn_data)
    
    # Fetch the neural network output
    inDet = detectionNN.get()
    if inDet is not None:
        detections = inDet.detections
        # Annotate the image if object is detected
        image_res = annotate_frame(image_res, detections)
        print("Detections",detections)
    
    # Write the image to the output path
    output_image_path = os.path.join(OUTPUT_IMAGES_YOLOv8n, os.path.basename(TEST_DATA))
    cv2.imwrite(output_image_path, image_res)
    print(f"[INFO] Processed {TEST_DATA} and saved to {output_image_path}")

    # Verify if the image was saved
    if os.path.exists(output_image_path):
        print(f"[INFO] Successfully saved the image at {output_image_path}")
    else:
        print(f"[ERROR] Failed to save the image at {output_image_path}")
