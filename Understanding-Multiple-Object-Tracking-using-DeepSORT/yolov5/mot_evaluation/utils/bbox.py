"""
2D MOT2016 Evaluation Toolkit
An python reimplementation of toolkit in
 2DMOT16(https://motchallenge.net/data/MOT16/)

This file computes bounding box overlap

(C) Han Shen(thushenhan@gmail.com), 2018-02
"""
import numpy as np


def bbox_overlap(ex_box, gt_box):
    ex_box = ex_box.reshape(-1, 4)
    gt_box = gt_box.reshape(-1, 4)
    paded_gt = np.tile(gt_box, [ex_box.shape[0], 1])
    insec = intersection(ex_box, paded_gt)

    uni = areasum(ex_box, paded_gt) - insec
    return insec / uni


def intersection(a, b):
    x = np.maximum(a[:, 0], b[:, 0])
    y = np.maximum(a[:, 1], b[:, 1])
    w = np.minimum(a[:, 2], b[:, 2]) - x
    h = np.minimum(a[:, 3], b[:, 3]) - y
    return np.maximum(w, 0) * np.maximum(h, 0)


def areasum(a, b):
    return (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + \
        (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
