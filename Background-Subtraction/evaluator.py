import argparse
import glob
import os
import time

import cv2
import numpy as np
import pybgs as bgs

ALGORITHMS_TO_EVALUATE = [
    (cv2.bgsegm.createBackgroundSubtractorGSOC, "GSoC", {}),
    (bgs.SuBSENSE, "SuBSENSE", {}),
]


# https://github.com/opencv/opencv_contrib/blob/master/modules/bgsegm/samples/evaluation.py
def contains_relevant_files(root):
    return os.path.isdir(os.path.join(root, "groundtruth")) and os.path.isdir(
        os.path.join(root, "input"),
    )


def find_relevant_dirs(root):
    relevant_dirs = []
    for d in sorted(os.listdir(root)):
        d = os.path.join(root, d)
        if os.path.isdir(d):
            if contains_relevant_files(d):
                relevant_dirs += [d]
            else:
                relevant_dirs += find_relevant_dirs(d)
    return relevant_dirs


def load_sequence(root):
    gt_dir, frames_dir = os.path.join(root, "groundtruth"), os.path.join(root, "input")
    gt = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    f = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))
    assert len(gt) == len(f)
    return gt, f


def evaluate_algorithm(gt, frames, algo, algo_arguments):
    # instantiate background subtraction model
    bgs = algo(**algo_arguments)
    mask = []
    # start time evaluation
    t_start = time.time()

    for i in range(len(gt)):
        # read frames
        frame = np.uint8(cv2.imread(frames[i], cv2.IMREAD_COLOR))
        # feed the frames into the model
        mask.append(bgs.apply(frame))

    average_duration = (time.time() - t_start) / len(gt)
    average_precision, average_recall, average_f1, average_accuracy = [], [], [], []

    # initiate iteration over GT frames
    for i in range(len(gt)):
        # get GT masks
        gt_mask = np.uint8(cv2.imread(gt[i], cv2.IMREAD_GRAYSCALE))
        # obtain region of interest
        roi = (gt_mask == 255) | (gt_mask == 0)
        if roi.sum() > 0:
            gt_answer, answer = gt_mask[roi], mask[i][roi]

            # calculate true positives, true negatives, false positives, false negatives
            tp = ((answer == 255) & (gt_answer == 255)).sum()
            tn = ((answer == 0) & (gt_answer == 0)).sum()
            fp = ((answer == 255) & (gt_answer == 0)).sum()
            fn = ((answer == 0) & (gt_answer == 255)).sum()

            # compute precision, recall, F1, accuracy to evaluate BS-model work
            if tp + fp > 0:
                average_precision.append(float(tp) / (tp + fp))
            if tp + fn > 0:
                average_recall.append(float(tp) / (tp + fn))
            if tp + fn + fp > 0:
                average_f1.append(2.0 * tp / (2.0 * tp + fn + fp))
            average_accuracy.append(float(tp + tn) / (tp + tn + fp + fn))

    return (
        average_duration,
        np.mean(average_precision),
        np.mean(average_recall),
        np.mean(average_f1),
        np.mean(average_accuracy),
    )


def evaluate_on_sequence(seq, summary):
    gt, frames = load_sequence(seq)
    category, video_name = os.path.basename(os.path.dirname(seq)), os.path.basename(seq)
    print("=== %s:%s ===" % (category, video_name))

    for algo, algo_name, algo_arguments in ALGORITHMS_TO_EVALUATE:
        print("Algorithm name: %s" % algo_name)
        sec_per_step, precision, recall, f1, accuracy = evaluate_algorithm(
            gt, frames, algo, algo_arguments,
        )
        print("Average accuracy: %.3f" % accuracy)
        print("Average precision: %.3f" % precision)
        print("Average recall: %.3f" % recall)
        print("Average F1: %.3f" % f1)
        print("Average sec. per step: %.4f" % sec_per_step)
        print("")

        if category not in summary:
            summary[category] = {}
        if algo_name not in summary[category]:
            summary[category][algo_name] = []
        summary[category][algo_name].append((precision, recall, f1, accuracy))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all background subtractors using Change Detection dataset",
    )
    parser.add_argument(
        "--dataset_path",
        help="Path to the directory with dataset. It may contain multiple inner directories. It will be scanned recursively.",
        required=True,
    )
    parser.add_argument("--algorithm", help="Test particular algorithm instead of all.")

    args = parser.parse_args()
    dataset_dirs = find_relevant_dirs(args.dataset_path)
    assert len(dataset_dirs) > 0, (
        "Passed directory must contain at least one sequence from the Change Detection dataset. There is no relevant directories in %s. Check that this directory is correct."
        % (args.dataset_path)
    )
    if args.algorithm is not None:
        global ALGORITHMS_TO_EVALUATE
        ALGORITHMS_TO_EVALUATE = filter(
            lambda a: a[1].lower() == args.algorithm.lower(), ALGORITHMS_TO_EVALUATE,
        )
    summary = {}

    for seq in dataset_dirs:
        evaluate_on_sequence(seq, summary)

    for category in summary:
        for algo_name in summary[category]:
            summary[category][algo_name] = np.mean(summary[category][algo_name], axis=0)

    algorithms_results = {
        "GSoC": [],
        "SuBSENSE": [],
    }

    for category in summary:
        print("=== SUMMARY for %s (Precision, Recall, F1, Accuracy) ===" % category)
        for algo_name in summary[category]:
            print(
                "%05s: %.3f %.3f %.3f %.3f"
                % ((algo_name,) + tuple(summary[category][algo_name])),
            )
            algorithms_results[algo_name].append(summary[category][algo_name])

    print("=== SUMMARY for all video categories (Precision, Recall, F1, Accuracy) ===")
    for algo_name in algorithms_results:
        algorithms_results[algo_name] = np.mean(
            np.array(algorithms_results[algo_name]), axis=0,
        )
        res_array = algorithms_results[algo_name]
        print(
            "{}: {:.3f}, {:.3f}, {:.3f}, {:.3f}".format(
                algo_name, res_array[0], res_array[1], res_array[2], res_array[3],
            ),
        )


if __name__ == "__main__":
    main()
