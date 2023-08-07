"""
Deep SORT with various object detection models.

USAGE:
python deep_sort_tracking.py 
python deep_sort_tracking.py --threshold 0.5 --imgsz 320
python deep_sort_tracking.py --threshold 0.5 --model fasterrcnn_resnet50_fpn_v2
                                                     fasterrcnn_resnet50_fpn
                                                     fasterrcnn_mobilenet_v3_large_fpn
                                                     fasterrcnn_mobilenet_v3_large_320_fpn
                                                     fcos_resnet50_fpn
                                                     ssd300_vgg16
                                                     ssdlite320_mobilenet_v3_large
                                                     retinanet_resnet50_fpn
                                                     retinanet_resnet50_fpn_v2
"""

import torch
import torchvision
import cv2
import os
import time
import argparse
import numpy as np

from torchvision.transforms import ToTensor
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import convert_detections, annotate
from coco_classes import COCO_91_CLASSES

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input', 
    default='input/mvmhat_1_1.mp4',
    help='path to input video',
)
parser.add_argument(
    '--imgsz', 
    default=None,
    help='image resize, 640 will resize images to 640x640',
    type=int
)
parser.add_argument(
    '--model',
    default='fasterrcnn_resnet50_fpn_v2',
    help='model name',
    choices=[
        'fasterrcnn_resnet50_fpn_v2',
        'fasterrcnn_resnet50_fpn',
        'fasterrcnn_mobilenet_v3_large_fpn',
        'fasterrcnn_mobilenet_v3_large_320_fpn',
        'fcos_resnet50_fpn',
        'ssd300_vgg16',
        'ssdlite320_mobilenet_v3_large',
        'retinanet_resnet50_fpn',
        'retinanet_resnet50_fpn_v2'
    ]
)
parser.add_argument(
    '--threshold',
    default=0.8,
    help='score threshold to filter out detections',
    type=float
)
parser.add_argument(
    '--embedder',
    default='mobilenet',
    help='type of feature extractor to use',
    choices=[
        "mobilenet",
        "torchreid",
        "clip_RN50",
        "clip_RN101",
        "clip_RN50x4",
        "clip_RN50x16",
        "clip_ViT-B/32",
        "clip_ViT-B/16"
    ]
)
parser.add_argument(
    '--show',
    action='store_true',
    help='visualize results in real-time on screen'
)
parser.add_argument(
    '--cls', 
    nargs='+',
    default=[1],
    help='which classes to track',
    type=int
)
args = parser.parse_args()

np.random.seed(42)

OUT_DIR = 'outputs'
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))

print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
print(f"Detector: {args.model}")
print(f"Re-ID embedder: {args.embedder}")

# Load model.
model = getattr(torchvision.models.detection, args.model)(weights='DEFAULT')
# Set model to evaluation mode.
model.eval().to(device)

# Initialize a SORT tracker object.
tracker = DeepSort(max_age=30, embedder=args.embedder)

VIDEO_PATH = args.input
cap = cv2.VideoCapture(VIDEO_PATH)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_fps = int(cap.get(5))
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
save_name = VIDEO_PATH.split(os.path.sep)[-1].split('.')[0]
# Define codec and create VideoWriter object.
out = cv2.VideoWriter(
    f"{OUT_DIR}/{save_name}_{args.model}_{args.embedder}.mp4", 
    cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, 
    (frame_width, frame_height)
)

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.
while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()
    if ret:
        if args.imgsz != None:
            resized_frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 
                (args.imgsz, args.imgsz)
            )
        else:
            resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to tensor and send it to device (cpu or cuda).
        frame_tensor = ToTensor()(resized_frame).to(device)

        start_time = time.time()
        # Feed frame to model and get detections.
        det_start_time = time.time()
        with torch.no_grad():
            detections = model([frame_tensor])[0]
        det_end_time = time.time()

        det_fps = 1 / (det_end_time - det_start_time)
    
        # Convert detections to Deep SORT format.
        detections = convert_detections(detections, args.threshold, args.cls)
    
        # Update tracker with detections.
        track_start_time = time.time()
        tracks = tracker.update_tracks(detections, frame=frame)
        track_end_time = time.time()
        track_fps = 1 / (track_end_time - track_start_time)

        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1

        print(f"Frame {frame_count}/{frames}", 
              f"Detection FPS: {det_fps:.1f},", 
              f"Tracking FPS: {track_fps:.1f}, Total FPS: {fps:.1f}")
        # Draw bounding boxes and labels on frame.
        if len(tracks) > 0:
            frame = annotate(
                tracks, 
                frame, 
                resized_frame,
                frame_width,
                frame_height,
                COLORS
            )
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (int(20), int(40)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255),
            thickness=2,
            lineType=cv2.LINE_AA
        )
        out.write(frame)
        if args.show:
            # Display or save output frame.
            cv2.imshow("Output", frame)
            # Press q to quit.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else:
        break
    
# Release resources.
cap.release()
cv2.destroyAllWindows()