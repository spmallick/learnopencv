import os
import cv2
import glob
import time
import torch
import argparse
import numpy as np
from tqdm.auto import tqdm

from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES

# ----------------------------------
# Argument parsing
# ----------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Path to the input video directory", required=True)
parser.add_argument(
    "--imgsz",
    default=None,
    type=int,
    help="Optional resize dimension (square). If set, each frame is resized to (imgsz, imgsz)",
)
parser.add_argument("--threshold", default=0.25, type=float, help="Detection threshold (score >= threshold)")
args = vars(parser.parse_args())

os.makedirs("inference_outputs/videos", exist_ok=True)

# ----------------------------------
# Fixed Colors (optional) or random
# ----------------------------------
# Example fixed colors for 5 classes (including background). Adjust as needed.
# COLORS = [
#     (0, 0, 255),     # Red      (class 1)
#     (147, 20, 255),  # Pink     (class 2)
#     (0, 255, 0),     # Green    (class 3)
#     (238, 130, 238), # Violet   (class 4)
#     (255, 255, 0),   # Cyan     (class 5)
# ]

# OR random colors:
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# ----------------------------------
# Load Model
# ----------------------------------
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load("outputs/best_model_79.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE).eval()

# ----------------------------------
# Gather video files
# ----------------------------------
video_dir = args["input"]
video_files = (
    glob.glob(os.path.join(video_dir, "*.mp4"))
    + glob.glob(os.path.join(video_dir, "*.avi"))
    + glob.glob(os.path.join(video_dir, "*.mov"))
)  # etc. if needed
print(f"Found {len(video_files)} video(s) in '{video_dir}'")

# Track total FPS across all frames of all videos
total_fps = 0.0
frame_count = 0

# ----------------------------------
# Process Each Video
# ----------------------------------
for vid_path in tqdm(video_files, desc="Videos"):
    # Extract just the base name for saving the output
    video_name = os.path.splitext(os.path.basename(vid_path))[0]
    out_path = os.path.join("inference_outputs", "videos", f"{video_name}_out.mp4")

    cap = cv2.VideoCapture(vid_path)
    if not cap.isOpened():
        print(f"Could not open {vid_path}. Skipping...")
        continue

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    if fps_cap <= 0:
        fps_cap = 20.0  # default if FPS can't be read

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 'XVID' also works
    out_writer = cv2.VideoWriter(out_path, fourcc, fps_cap, (width, height))

    # For progress bar of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar_frames = tqdm(total=total_frames, desc=f"Frames of {video_name}", leave=False)

    # Read frames in a loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar_frames.update(1)
        orig_frame = frame.copy()

        # Optional resizing if imgsz is set
        if args["imgsz"] is not None:
            frame = cv2.resize(frame, (args["imgsz"], args["imgsz"]))

        # Pre-process
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
        frame_rgb /= 255.0
        frame_tensor = torch.tensor(frame_rgb.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0).to(DEVICE)

        # Inference
        start_time = time.time()
        with torch.no_grad():
            outputs = model(frame_tensor)
        end_time = time.time()

        # Calculate FPS for this frame
        frame_fps = 1 / (end_time - start_time)
        total_fps += frame_fps
        frame_count += 1

        # Move detections to CPU
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        boxes = outputs[0]["boxes"].numpy()
        scores = outputs[0]["scores"].numpy()
        labels = outputs[0]["labels"].numpy().astype(int)

        # Filter by confidence threshold
        valid_idx = np.where(scores >= args["threshold"])[0]
        boxes = boxes[valid_idx].astype(int)
        labels = labels[valid_idx]

        # If frame was resized for inference, rescale boxes back to orig size
        if args["imgsz"] is not None:
            w_new, h_new = args["imgsz"], args["imgsz"]
            h_orig, w_orig = orig_frame.shape[:2]
            # scale boxes from [0..w_new/h_new] to [0..w_orig/h_orig]
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] / w_new) * w_orig
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] / h_new) * h_orig
            boxes = boxes.astype(int)

        # Draw bounding boxes
        for (x1, y1, x2, y2), lab in zip(boxes, labels):
            class_name = CLASSES[lab]
            color = COLORS[lab % len(CLASSES)]  # (B, G, R)
            cv2.rectangle(orig_frame, (x1, y1), (x2, y2), color[::-1], 2)
            cv2.putText(
                orig_frame,
                class_name,
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color[::-1],
                2,
                lineType=cv2.LINE_AA,
            )

        # Write the processed frame to the output video
        out_writer.write(orig_frame)

    # Close out everything for this video
    pbar_frames.close()
    cap.release()
    out_writer.release()
    print(f"Processed video saved at: {out_path}")

# ----------------------------------
# Print Overall FPS
# ----------------------------------
if frame_count > 0:
    avg_fps = total_fps / frame_count
    print(f"Overall Average FPS across all videos: {avg_fps:.3f}")

print("VIDEO INFERENCE COMPLETE!")
