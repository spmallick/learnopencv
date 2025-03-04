import os
import cv2
import time
import torch
import gradio as gr
import numpy as np

# Make sure these are your local imports from your project.
from model import create_model
from config import NUM_CLASSES, DEVICE, CLASSES

# ----------------------------------------------------------------
# GLOBAL SETUP
# ----------------------------------------------------------------
# Create the model and load the best weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load("outputs/best_model_79.pth", map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE).eval()

# Create a colors array for each class index.
# (length matches len(CLASSES), including background if you wish).
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# COLORS = [
#     (255, 255, 0),  # Cyan - background
#     (50, 0, 255),  # Red - buffalo
#     (147, 20, 255),  # Pink - elephant
#     (0, 255, 0),  # Green - rhino
#     (238, 130, 238),  # Violet - zebra
# ]


# ----------------------------------------------------------------
# HELPER FUNCTIONS
# ----------------------------------------------------------------
def inference_on_image(orig_image: np.ndarray, resize_dim=None, threshold=0.25):
    """
    Runs inference on a single image (OpenCV BGR or NumPy array).
    - resize_dim: if not None, we resize to (resize_dim, resize_dim)
    - threshold: detection confidence threshold
    Returns: processed image with bounding boxes drawn.
    """
    image = orig_image.copy()
    # Optionally resize for inference.
    if resize_dim is not None:
        image = cv2.resize(image, (resize_dim, resize_dim))

    # Convert BGR to RGB, normalize [0..1]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # Move channels to front (C,H,W)
    image_tensor = torch.tensor(image_rgb.transpose(2, 0, 1), dtype=torch.float).unsqueeze(0).to(DEVICE)
    start_time = time.time()
    # Inference
    with torch.no_grad():
        outputs = model(image_tensor)
    end_time = time.time()
    # Get the current fps.
    fps = 1 / (end_time - start_time)
    fps_text = f"FPS: {fps:.2f}"
    # Move outputs to CPU numpy
    outputs = [{k: v.cpu() for k, v in t.items()} for t in outputs]
    boxes = outputs[0]["boxes"].numpy()
    scores = outputs[0]["scores"].numpy()
    labels = outputs[0]["labels"].numpy().astype(int)

    # Filter out boxes with low confidence
    valid_idx = np.where(scores >= threshold)[0]
    boxes = boxes[valid_idx].astype(int)
    labels = labels[valid_idx]

    # If we resized for inference, rescale boxes back to orig_image size
    if resize_dim is not None:
        h_orig, w_orig = orig_image.shape[:2]
        h_new, w_new = resize_dim, resize_dim
        # scale boxes
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] / w_new) * w_orig
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] / h_new) * h_orig

    # Draw bounding boxes
    for box, label_idx in zip(boxes, labels):
        class_name = CLASSES[label_idx] if 0 <= label_idx < len(CLASSES) else str(label_idx)
        color = COLORS[label_idx % len(COLORS)][::-1]  # BGR color
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), color, 5)
        cv2.putText(orig_image, class_name, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(
            orig_image,
            fps_text,
            (int((w_orig / 2) - 50), 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return orig_image, fps


def inference_on_frame(frame: np.ndarray, resize_dim=None, threshold=0.25):
    """
    Same as inference_on_image but for a single video frame.
    Returns the processed frame with bounding boxes.
    """
    return inference_on_image(frame, resize_dim, threshold)


# ----------------------------------------------------------------
# GRADIO FUNCTIONS
# ----------------------------------------------------------------


def img_inf(image_path, resize_dim, threshold):
    """
    Gradio function for image inference.
    :param image_path: File path from Gradio (uploaded image).
    :param model_name: Selected model from Radio (not used if only one model).
    Returns: A NumPy image array with bounding boxes.
    """
    if image_path is None:
        return None  # No image provided
    orig_image = cv2.imread(image_path)  # BGR
    if orig_image is None:
        return None  # Error reading image

    result_image, _ = inference_on_image(orig_image, resize_dim=resize_dim, threshold=threshold)
    # Return the image in RGB for Gradio's display
    result_image_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
    return result_image_rgb


def vid_inf(video_path, resize_dim, threshold):
    """
    Gradio function for video inference.
    Processes each frame, draws bounding boxes, and writes to an output video.
    Returns: (last_processed_frame, output_video_file_path)
    """
    if video_path is None:
        return None, None  # No video provided

    # Prepare input capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, None

    # Create an output file path
    os.makedirs("inference_outputs/videos", exist_ok=True)
    out_video_path = os.path.join("inference_outputs/videos", "video_output.mp4")
    # out_video_path = "video_output.mp4"

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # or 'XVID'

    # If FPS is 0 (some weird container), default to something
    if fps <= 0:
        fps = 20.0

    out_writer = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    frame_count = 0
    total_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Inference on frame
        processed_frame, frame_fps = inference_on_frame(frame, resize_dim=resize_dim, threshold=threshold)
        total_fps += frame_fps
        frame_count += 1

        # Write the processed frame
        out_writer.write(processed_frame)
        yield cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), None

    avg_fps = total_fps / frame_count

    cap.release()
    out_writer.release()
    print(f"Average FPS: {avg_fps:.3f}")
    yield None, out_video_path


# ----------------------------------------------------------------
# BUILD THE GRADIO INTERFACES
# ----------------------------------------------------------------

# For demonstration, we define two possible model radio choices.
# You can ignore or expand this if you only use RetinaNet.
resize_dim = gr.Slider(100, 1024, value=640, label="Resize Dimension", info="Resize image to this dimension")
threshold = gr.Slider(0, 1, value=0.5, label="Threshold", info="Confidence threshold for detection")
inputs_image = gr.Image(type="filepath", label="Input Image")
outputs_image = gr.Image(type="numpy", label="Output Image")

interface_image = gr.Interface(
    fn=img_inf,
    inputs=[inputs_image, resize_dim, threshold],
    outputs=outputs_image,
    title="Image Inference",
    description="Upload your photo, select a model, and see the results!",
    examples=[["examples/buffalo.jpg"], ["examples/zebra.jpg"]],
    cache_examples=False,
)

resize_dim = gr.Slider(100, 1024, value=640, label="Resize Dimension", info="Resize image to this dimension")
threshold = gr.Slider(0, 1, value=0.5, label="Threshold", info="Confidence threshold for detection")
input_video = gr.Video(label="Input Video")

# Output is a pair: (last_processed_frame, output_video_path)
output_frame = gr.Image(type="numpy", label="Output (Last Processed Frame)")
output_video_file = gr.Video(format="mp4", label="Output Video")

interface_video = gr.Interface(
    fn=vid_inf,
    inputs=[input_video, resize_dim, threshold],
    outputs=[output_frame, output_video_file],
    title="Video Inference",
    description="Upload your video and see the processed output!",
    examples=[["examples/elephants.mp4"], ["examples/rhino.mp4"]],
    cache_examples=False,
)

# Combine them in a Tabbed Interface
demo = (
    gr.TabbedInterface(
        [interface_image, interface_video],
        tab_names=["Image", "Video"],
        title="FineTuning RetinaNet for Wildlife Animal Detection",
        theme="gstaff/xkcd",
    )
    .queue()
    .launch()
)
