"""
2D MOT2016 Evaluation Toolkit
An python reimplementation of toolkit in 
2DMOT16(https://motchallenge.net/data/MOT16/)

This file deals with file IO / invalid annotation
 removal / result output

(C) Han Shen(thushenhan@gmail.com), 2018-02
"""
import os
import numpy as np


def read_seqmaps(fname):
    """
    seqmap: list the sequence name to be evaluated
    """
    assert os.path.exists(fname), 'File %s not exists!' % fname
    with open(fname, 'r') as fid:
        lines = [line.strip() for line in fid.readlines()]
        seqnames = lines[1:]
    return seqnames


def read_txt_to_struct(fname):
    """
    read txt to structure, the column represents:
    [frame number] [identity number] [bbox left] [bbox top]
     [bbox width] [bbox height] [DET: detection score,
      GT: ignored class flag] [class] [visibility ratio]
    """
    data = []
    with open(fname, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = list(map(float, line.strip().split(',')))
            data.append(line)
    data = np.array(data)
    # change point-size format to two-points format
    data[:, 4:6] += data[:, 2:4]
    return data


def extract_valid_gt_data(all_data, remove_ofv=False):
    """
    remove non-valid classes. 
    following mot2016 format,
     valid class include [1: pedestrain],
     distractor classes include [2: person on vehicle, 
      7: static person, 8: distractor, 12: reflection].
    """
    distractor_classes = [2, 7, 8, 12]
    valid_classes = [1]
    original = all_data.shape[0]
    # remove classes in other classes, pedestrain and distractors
    # left for furthur usages
    selected = np.array([
        i for i in range(all_data.shape[0])
        if all_data[i, 7] in valid_classes + distractor_classes])
    all_data = all_data[selected, :]
    # remove boxes whose centers is out of view
    # Cause this tool is not only set for MOT, thus resolution is not assumed
    #  provided. In MOT, the maximum width andd height should be taken into
    #  consirderation

    # PS: As stated by author of MOT benchmark, it would be better the tracker
    #  could figure out the out of view pedestrain like human does. Thus no
    #  filtering
    if remove_ofv:
        selected = np.array([i for i in range(all_data.shape[0])
                             if (all_data[i, 2] + all_data[i, 4]) / 2 >= 0 and
                             (all_data[i, 3] + all_data[i, 5]) / 2 >= 0])
        # not consider right and bottom out of range here. Anyway ofv is not
        # removed in MOT2016
        # selected = np.array([i for i in xrange(all_data.shape[0])
        #                       if (all_data[i, 2] + all_data[i, 4]) / 2 != 0
        #                          ])
        all_data = all_data[selected, :]

    # remove non-human classes from ground truth, and return distractor
    #  identities
    cond = np.array(
        [i in valid_classes + distractor_classes for i in all_data[:, 7]])
    selected = np.where(cond == True)[0]
    all_data = all_data[selected, :]

    print('[GT PREPROCESSING]: Removing non-people classes, remaining '
          '{}/{} boxes'.format(all_data.shape[0], original))
    cond = np.array([i in distractor_classes for i in all_data[:, 7]])
    selected = np.where(cond == True)[0]
    distractor_ids = np.unique(all_data[selected, 1])
    return all_data, distractor_ids


def print_format(widths, formaters, values, form_attr):
    return ' '.join([(form_attr % (width, form)).format(val) for (
        form, width, val) in zip(formaters, widths, values)])


def print_format_name(widths, values, form_attr):
    return ' '.join([(form_attr % (width)).format(val) for (width, val) in zip(
        widths, values)])


def print_metrics(header, metrics, banner=25):
    if len(metrics) == 17:
        print_metrics_ext(header, metrics)
        return
    print('\n', '*' * banner, header, '*' * banner)
    # metric_names_long = ['Recall', 'Precision', 'False Alarm Rate',
    #                      'GT Tracks', 'Mostly Tracked', 'Partially Tracked',
    #                      'Mostly Lost', 'False Positives', 'False Negatives',
    #                      'ID Switches', 'Fragmentations',
    #                      'MOTA', 'MOTP', 'MOTA Log']

    metric_names_short = ['Rcll', 'Prcn', 'FAR',
                          'GT', 'MT', 'PT', 'ML',
                          'FP', 'FN', 'IDs', 'FM',
                          'MOTA', 'MOTP', 'MOTAL']

    # metric_widths_long = [6, 9, 16, 9, 14, 17, 11, 15, 15, 11, 14, 5, 5, 8]
    metric_widths_short = [5, 5, 5, 4, 4, 4, 4, 6, 6, 5, 5, 5, 5, 5]

    metric_format_long = ['.1f', '.1f', '.2f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.1f', '.1f', '.1f']

    splits = [(0, 3), (3, 7), (7, 11), (11, 14)]
    print(' | '.join([print_format_name(
                     metric_widths_short[start:end],
                     metric_names_short[start:end], '{0: <%d}')
        for (start, end) in splits]))

    print(' | '.join([print_format(
                     metric_widths_short[start:end],
                     metric_format_long[start:end],
                     metrics[start:end], '{:%d%s}')
        for (start, end) in splits]))


def print_metrics_ext(header, metrics, banner=30):
    print('\n{} {} {}'.format('*' * banner, header, '*' * banner))
    # metric_names_long = ['IDF1', 'IDP', 'IDR',
    #                      'Recall', 'Precision', 'False Alarm Rate',
    #                      'GT Tracks', 'Mostly Tracked', 'Partially Tracked',
    #                      'Mostly Lost',
    #                      'False Positives', 'False Negatives', 'ID Switches',
    #                      'Fragmentations',
    #                      'MOTA', 'MOTP', 'MOTA Log']

    metric_names_short = ['IDF1', 'IDP', 'IDR',
                          'Rcll', 'Prcn', 'FAR',
                          'GT', 'MT', 'PT', 'ML',
                          'FP', 'FN', 'IDs', 'FM',
                          'MOTA', 'MOTP', 'MOTAL']

    # metric_widths_long = [5, 4, 4, 6, 9, 16,
    #   9, 14, 17, 11, 15, 15, 11, 14, 5, 5, 8]
    metric_widths_short = [5, 4, 4, 5, 5, 5, 4, 4, 4, 4, 6, 6, 5, 5, 5, 5, 5]

    metric_format_long = ['.1f', '.1f', '.1f',
                          '.1f', '.1f', '.2f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.0f', '.0f', '.0f', '.0f',
                          '.1f', '.1f', '.1f']

    splits = [(0, 3), (3, 6), (6, 10), (10, 14), (14, 17)]

    print(' | '.join([print_format_name(
                     metric_widths_short[start:end],
                     metric_names_short[start:end], '{0: <%d}')
        for (start, end) in splits]))

    print(' | '.join([print_format(
                     metric_widths_short[start:end],
                     metric_format_long[start:end],
                     metrics[start:end], '{:%d%s}')
        for (start, end) in splits]))
    print('\n\n')