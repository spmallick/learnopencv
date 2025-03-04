import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# If you have a custom nms function, you can import or define here:
# from custom_utils import non_max_suppression  # example placeholder


# For demonstration, let's define a basic IoU-based NMS.
def nms(boxes, scores, iou_threshold=0.5):
    """
    Basic NMS implementation.
    boxes: Nx4 numpy array [x1, y1, x2, y2]
    scores: Nx1 numpy array
    iou_threshold: float
    returns: indices of boxes to keep
    """
    # Adapted from common NMS logic
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        current = idxs[0]
        keep.append(current)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[current], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep


def compute_iou(boxA, boxes):
    """
    boxA: [x1,y1,x2,y2], boxes: Nx4
    returns: IoU array of length N
    """
    # Coordinates
    x1 = np.maximum(boxA[0], boxes[:, 0])
    y1 = np.maximum(boxA[1], boxes[:, 1])
    x2 = np.minimum(boxA[2], boxes[:, 2])
    y2 = np.minimum(boxA[3], boxes[:, 3])

    interArea = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    iou = interArea / (boxAArea + boxBArea - interArea + 1e-6)
    return iou


def preprocess_frame(frame, input_size=640):
    """
    Resizes frame to (input_size, input_size),
    converts BGR->RGB, normalizes, expands dims to (1,3,H,W).
    """
    frame_resized = cv2.resize(frame, (input_size, input_size))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
    # If your training used [0..1] or mean/std, do so here
    frame_rgb /= 255.0
    # shape (H,W,3) -> (1,3,H,W)
    frame_input = np.transpose(frame_rgb, (2, 0, 1))[np.newaxis, ...]  # float32
    return frame_input


def postprocess(boxes, scores, labels, frame_shape, input_size=640, score_threshold=0.5, iou_threshold=0.5):
    """
    Applies threshold, NMS, and rescales boxes to original frame size.
    """
    # 1) Filter by confidence threshold
    keep_idx = scores >= score_threshold
    boxes = boxes[keep_idx]
    scores = scores[keep_idx]
    labels = labels[keep_idx]

    # 2) NMS
    keep_nms = nms(boxes, scores, iou_threshold)
    boxes = boxes[keep_nms]
    scores = scores[keep_nms]
    labels = labels[keep_nms]

    # 3) Rescale boxes from input_size back to original frame size
    H, W = frame_shape[:2]
    scale_w, scale_h = W / input_size, H / input_size
    boxes[:, [0, 2]] *= scale_w
    boxes[:, [1, 3]] *= scale_h

    return boxes, scores, labels


def draw_boxes(frame, boxes, scores, labels, class_names=None):
    """
    Draws bounding boxes on the frame.
    class_names: list of class strings, e.g. ["__background__", "elephant", ...]
    """
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        score = scores[i]
        class_id = int(labels[i])
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        text = f"{class_names[class_id] if class_names else class_id}: {score:.2f}"
        cv2.putText(frame, text, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)


def run_onnx_inference_video(
    video_path,
    onnx_path="outputs/retinanet.onnx",
    output_path="outputs/output_video.mp4",
    input_size=640,
    score_threshold=0.5,
    iou_threshold=0.5,
    class_names=None,
):
    """
    Runs inference on a video using an ONNX model for RetinaNet-like output.
    Post-processes with threshold + NMS, then draws boxes on frames.
    Saves result to a new video file.
    """
    # 1) Initialize ONNX runtime session
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    # 2) Capture input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Could not open video: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 3) Preprocess frame
        frame_input = preprocess_frame(frame, input_size=input_size)  # shape: (1,3,H,W)

        # 4) ONNX inference
        #    The exported model (x) returns [boxes, scores, labels]
        onnx_inputs = {"images": frame_input}
        outputs = ort_session.run(None, onnx_inputs)
        # outputs is a list, e.g. [boxes array, scores array, labels array]
        boxes_onnx, scores_onnx, labels_onnx = outputs
        # print("score", scores_onnx)
        # 5) Post-process (threshold, NMS, rescaling)
        boxes_np, scores_np, labels_np = postprocess(
            boxes_onnx,  # shape (N,4)
            scores_onnx,  # shape (N,)
            labels_onnx,  # shape (N,)
            frame.shape,
            input_size=input_size,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
        )

        # 6) Draw detection results
        draw_boxes(frame, boxes_np, scores_np, labels_np, class_names)

        # 7) Write to output video
        out_writer.write(frame)

        frame_count += 1

    cap.release()
    out_writer.release()
    end_time = time.time()
    total_time = end_time - start_time
    fps = frame_count / total_time
    print(f"Inference completed. Processed {frame_count} frames in {total_time:.2f}s at {fps:.2f} fps.")


if __name__ == "__main__":
    # Example usage:
    # python infer_onnx_video.py --video_path sample_video.mp4
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True, help="Path to input video.")
    parser.add_argument("--onnx_path", type=str, default="outputs/retinanet.onnx", help="Path to ONNX model.")
    parser.add_argument(
        "--output_path", type=str, default="outputs/output_onnx_video.mp4", help="Output video file path."
    )
    parser.add_argument("--input_size", type=int, default=640, help="Input image size for model.")
    parser.add_argument("--score_threshold", type=float, default=0.7, help="Confidence threshold.")
    parser.add_argument("--iou_threshold", type=float, default=0.5, help="IoU threshold for NMS.")
    args = parser.parse_args()

    class_names = ["__background__", "buffalo", "elephant", "rhino", "zebra"]

    run_onnx_inference_video(
        video_path=args.video_path,
        onnx_path=args.onnx_path,
        output_path=args.output_path,
        input_size=args.input_size,
        score_threshold=args.score_threshold,
        iou_threshold=args.iou_threshold,
        class_names=class_names,
    )
