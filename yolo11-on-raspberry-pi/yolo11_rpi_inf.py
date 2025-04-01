import cv2
import numpy as np
import time
from ultralytics import YOLO
from collections import defaultdict


def inference(
    model,
    mode,
    task,
    video_path=None,
    save_output=False,
    output_path="output.mp4",
    show_output=True,
    count=False,
    show_tracks=False,
):
    if mode == "cam":
        cap = cv2.VideoCapture(0)
    elif mode == "video":
        if video_path is None:
            raise ValueError("Please provide a valid video path for video mode.")
        cap = cv2.VideoCapture(video_path)
    else:
        raise ValueError("Invalid mode. Use 'cam' or 'video'.")

    # History for tracking lines
    track_history = defaultdict(lambda: [])

    # History for unique object IDs per class (used in tracking count)
    seen_ids_per_class = defaultdict(set)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    out = None

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Failed to read frame or end of video")
            break

        start_time = time.time()
        class_counts = defaultdict(int)

        # Inference
        if task == "track":
            results = model.track(frame, conf=0.3, persist=True, tracker="bytetrack.yaml")
        elif task == "detect":
            results = model.predict(frame, conf=0.5)
        else:
            raise ValueError("Invalid task. Use 'detect' or 'track'.")

        end_time = time.time()
        annotated_frame = results[0].plot()

        if results[0].boxes and results[0].boxes.cls is not None:
            boxes = results[0].boxes.xywh.cpu()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            names = results[0].names

            if task == "track" and results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, cls_id, track_id in zip(boxes, class_ids, track_ids):
                    x, y, w, h = box
                    class_name = names[cls_id]

                    # Save this ID for unique counting
                    if count:
                        seen_ids_per_class[class_name].add(track_id)

                    # Draw tracking lines
                    if show_tracks:
                        track = track_history[track_id]
                        track.append((float(x), float(y)))
                        if len(track) > 30:
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            elif task == "detect" and count:
                for cls_id in class_ids:
                    class_counts[names[cls_id]] += 1

        # Draw class counts in bottom-left corner
        if count:
            x0, y0 = 10, annotated_frame.shape[0] - 80
            if task == "track":
                for i, (cls_name, ids) in enumerate(seen_ids_per_class.items()):
                    label = f"{cls_name}: {len(ids)}"
                    y = y0 + i * 25
                    cv2.putText(annotated_frame, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            elif task == "detect":
                for i, (cls_name, total) in enumerate(class_counts.items()):
                    label = f"{cls_name}: {total}"
                    y = y0 + i * 25
                    cv2.putText(annotated_frame, label, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Draw FPS
        fps = 1 / (end_time - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if save_output:
            if out is None:
                height, width = annotated_frame.shape[:2]
                out = cv2.VideoWriter(output_path, fourcc, input_fps, (width, height))
            out.write(annotated_frame)

        if show_output:
            cv2.imshow("Raspbery Pi x YOLO11 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()


# Example usage
model = YOLO("yolo11n_mnn_model", task="detect")
# model = YOLO("yolo11n-seg_openvino_model", task="segment")
# model = YOLO("yolo11n-pose_ncnn_model", task="pose")


inference(
    model,
    mode="cam",
    task="track",
    video_path="baseball.mov",
    save_output=True,
    show_output=True,
    count=True,
    show_tracks=False,
)
