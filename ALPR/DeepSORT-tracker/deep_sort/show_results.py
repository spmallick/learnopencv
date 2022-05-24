# vim: expandtab:ts=4:sw=4
import argparse

import cv2
import numpy as np

import deep_sort_app
from deep_sort.iou_matching import iou
from application_util import visualization


DEFAULT_UPDATE_MS = 20


def run(sequence_dir, result_file, show_false_alarms=False, detection_file=None,
        update_ms=None, video_filename=None):
    """Run tracking result visualization.

    Parameters
    ----------
    sequence_dir : str
        Path to the MOTChallenge sequence directory.
    result_file : str
        Path to the tracking output file in MOTChallenge ground truth format.
    show_false_alarms : Optional[bool]
        If True, false alarms are highlighted as red boxes.
    detection_file : Optional[str]
        Path to the detection file.
    update_ms : Optional[int]
        Number of milliseconds between cosecutive frames. Defaults to (a) the
        frame rate specifid in the seqinfo.ini file or DEFAULT_UDPATE_MS ms if
        seqinfo.ini is not available.
    video_filename : Optional[Str]
        If not None, a video of the tracking results is written to this file.

    """
    seq_info = deep_sort_app.gather_sequence_info(sequence_dir, detection_file)
    results = np.loadtxt(result_file, delimiter=',')

    if show_false_alarms and seq_info["groundtruth"] is None:
        raise ValueError("No groundtruth available. Cannot show false alarms.")

    def frame_callback(vis, frame_idx):
        print("Frame idx", frame_idx)
        image = cv2.imread(
            seq_info["image_filenames"][frame_idx], cv2.IMREAD_COLOR)

        vis.set_image(image.copy())

        if seq_info["detections"] is not None:
            detections = deep_sort_app.create_detections(
                seq_info["detections"], frame_idx)
            vis.draw_detections(detections)

        mask = results[:, 0].astype(np.int) == frame_idx
        track_ids = results[mask, 1].astype(np.int)
        boxes = results[mask, 2:6]
        vis.draw_groundtruth(track_ids, boxes)

        if show_false_alarms:
            groundtruth = seq_info["groundtruth"]
            mask = groundtruth[:, 0].astype(np.int) == frame_idx
            gt_boxes = groundtruth[mask, 2:6]
            for box in boxes:
                # NOTE(nwojke): This is not strictly correct, because we don't
                # solve the assignment problem here.
                min_iou_overlap = 0.5
                if iou(box, gt_boxes).max() < min_iou_overlap:
                    vis.viewer.color = 0, 0, 255
                    vis.viewer.thickness = 4
                    vis.viewer.rectangle(*box.astype(np.int))

    if update_ms is None:
        update_ms = seq_info["update_ms"]
    if update_ms is None:
        update_ms = DEFAULT_UPDATE_MS
    visualizer = visualization.Visualization(seq_info, update_ms)
    if video_filename is not None:
        visualizer.viewer.enable_videowriter(video_filename)
    visualizer.run(frame_callback)


def parse_args():
    """ Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Siamese Tracking")
    parser.add_argument(
        "--sequence_dir", help="Path to the MOTChallenge sequence directory.",
        default=None, required=True)
    parser.add_argument(
        "--result_file", help="Tracking output in MOTChallenge file format.",
        default=None, required=True)
    parser.add_argument(
        "--detection_file", help="Path to custom detections (optional).",
        default=None)
    parser.add_argument(
        "--update_ms", help="Time between consecutive frames in milliseconds. "
        "Defaults to the frame_rate specified in seqinfo.ini, if available.",
        default=None)
    parser.add_argument(
        "--output_file", help="Filename of the (optional) output video.",
        default=None)
    parser.add_argument(
        "--show_false_alarms", help="Show false alarms as red bounding boxes.",
        type=bool, default=False)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.sequence_dir, args.result_file, args.show_false_alarms,
        args.detection_file, args.update_ms, args.output_file)
