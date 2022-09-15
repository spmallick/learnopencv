#!/usr/bin/env python3
"""
 Copyright (C) 2018-2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import colorsys
import logging
import random
import sys
import os
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
from yolo_to_ssd_classes import yolo_to_ssd_classes

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))


import models
import monitors
from pipelines import AsyncPipeline
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics



logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

import json


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-at', '--architecture_type', help='Required. Specify model\' architecture type.',
                      type=str, required=True, choices=('ssd', 'yolo', 'yolov4', 'faceboxes', 'centernet', 'ctpn', 'retinaface'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                                   help='Optional. Keeps aspect ratio on resize.')
    common_model_args.add_argument('--input_size', default=(600, 600), type=int, nargs=2,
                                   help='Optional. The first image size used for CTPN model reshaping. '
                                        'Default: 600 600. Note that submitted images should have the same resolution, '
                                        'otherwise predictions might be incorrect.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=1, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of output to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


def get_model(ie, args):
    if args.architecture_type == 'ssd':
        return models.SSD(ie, args.model, labels=args.labels, keep_aspect_ratio_resize=args.keep_aspect_ratio)
    elif args.architecture_type == 'ctpn':
        return models.CTPN(ie, args.model, input_size=args.input_size, threshold=args.prob_threshold)
    elif args.architecture_type == 'yolo':
        return models.YOLO(ie, args.model, labels=args.labels,
                           threshold=args.prob_threshold, keep_aspect_ratio=args.keep_aspect_ratio)
    elif args.architecture_type == 'yolov4':
        return models.YoloV4(ie, args.model, labels=args.labels,
                             threshold=args.prob_threshold, keep_aspect_ratio=args.keep_aspect_ratio)
    elif args.architecture_type == 'faceboxes':
        return models.FaceBoxes(ie, args.model, threshold=args.prob_threshold)
    elif args.architecture_type == 'centernet':
        return models.CenterNet(ie, args.model, labels=args.labels, threshold=args.prob_threshold)
    elif args.architecture_type == 'retinaface':
        return models.RetinaFace(ie, args.model, threshold=args.prob_threshold)
    else:
        raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(args.architecture_type))


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


def draw_detections(frame, detections, palette, labels, threshold, id):
    id = str(id.strip('0').split('.')[0])
    size = frame.shape[:2]
    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            color = palette[class_id]
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if isinstance(detection, models.DetectionWithLandmarks):
                for landmark in detection.landmarks:
                    cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 255), 2)

        # create_json(detection, size, id)
    return frame


def print_raw_results(size, detections, labels, threshold, id):
    id = str(id.strip('0').split('.')[0])
    log.info(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        if detection.score > threshold:
            
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            log.info('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                     .format(det_label, detection.score, xmin, ymin, xmax, ymax))
        


def main():
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    log.info('Loading network...')

    model = get_model(ie, args)

    detector_pipeline = AsyncPipeline(ie, model, plugin_config,
                                      device=args.device, max_num_requests=args.num_infer_requests)

    ### READ TIME ###
    read_time_start = time.time()
    cap = open_images_capture(args.input, args.loop)
    read_time_end = time.time()

    next_frame_id = 0
    next_frame_id_to_show = 0

    image_id = 0

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    palette = ColorPalette(len(model.labels) if model.labels else 100)
    metrics = PerformanceMetrics()
    presenter = None
    video_writer = cv2.VideoWriter()

    results_list = []
    detection_ids = [1, 3, 4]

    all_starts = 0
    
    while True:
        print('NEXT FRAME ID', next_frame_id)

        id = images[image_id]


        if next_frame_id == 5000:
            break


        if detector_pipeline.callback_exceptions:
            raise detector_pipeline.callback_exceptions[0]
        # Process all completed requests
        #### DETECTION TIME ####
        detect_time_start = time.time()
        results = detector_pipeline.get_result(next_frame_id_to_show)
        detect_time_end = time.time()
        detect_time_list.append(detect_time_end-detect_time_start)
        if results:
            objects, frame_meta = results
            
            for detection in objects:
                x = float(detection.xmin)
                y = float(detection.ymin)
                w = float(detection.xmax - detection.xmin)
                h = float(detection.ymax - detection.ymin)
                cls = detection.id
                cls = yolo_to_ssd_classes[cls]
                id = str(id.lstrip('0').split('.')[0])
                conf = detection.score
                # if cls in detection_ids:
                results_list.append({'image_id': int(id),
                                    'category_id': cls,
                                    'bbox': [x, y, w, h],
                                    'score': float(conf)})

            frame = frame_meta['frame']
            post_process_start = time.time()
            start_time = frame_meta['start_time']
            all_starts += start_time
            
            all_starts += start_time

            if len(objects) and args.raw_output_message:
                print_raw_results(frame.shape[:2], objects, model.labels, args.prob_threshold, images[image_id])

            presenter.drawGraphs(frame)
            frame = draw_detections(frame, objects, palette, model.labels, args.prob_threshold, images[image_id])

            metrics.update(start_time, frame)

            post_process_end = time.time()
            post_process_list.append(post_process_end-post_process_start)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)

            if not args.no_show:
                # cv2.imshow('Detection Results', frame)
                cv2.imwrite(f"/home/sovit/my_data/Data_Science/Projects/openvino_experiments/model_quantization/data/images/image_{image_id}.jpg", frame)
                # key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                #if key in {ord('q'), ord('Q'), ESC_KEY}:
                    #break
                #presenter.handleKey(key)
            next_frame_id_to_show += 1
            image_id += 1
            continue

        if detector_pipeline.is_ready():
            # Get new image/frame
            pre_process_start = time.time()
            start_time = perf_counter()
            frame = cap.read()

            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if next_frame_id == 0:
                presenter = monitors.Presenter(args.utilization_monitors, 55,
                                               (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))
                if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                         cap.fps(), (frame.shape[1], frame.shape[0])):
                    raise RuntimeError("Can't open video writer")
            # Submit for inference
            detector_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            pre_process_end = time.time()
            pre_process_list.append(pre_process_end-pre_process_start)
            next_frame_id += 1

        else:
            # Wait for empty request
            detector_pipeline.await_any()

    results_file = 'results.json'
    with open(results_file, 'w') as f:
            f.write(json.dumps(results_list, indent=4))

    detector_pipeline.await_all()
    # Process completed requests
    while detector_pipeline.has_completed_request():
        results = detector_pipeline.get_result(next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            post_process_two_start = time.time()
            start_time = frame_meta['start_time']

            if len(objects) and args.raw_output_message:
                print()
                # print_raw_results(frame.shape[:2], objects, model.labels, args.prob_threshold)

            presenter.drawGraphs(frame)
            # frame = draw_detections(frame, objects, palette, model.labels, args.prob_threshold)
            metrics.update(start_time, frame)
            post_process_two_end = time.time()
            post_process_list_two.append(post_process_two_end-post_process_two_start)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)
	    
            if not args.no_show:
                # cv2.imshow('Detection Results', frame)
                cv2.imwrite(f"/home/sovit/my_data/Data_Science/Projects/openvino_experiments/model_quantization/data/images/image_{frame_id}.jpg", frame)
                # key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)
            next_frame_id_to_show += 1
        else:
            break

    metrics.print_total()
    print("Presentor", presenter.reportMeans())


if __name__ == '__main__':
    images = os.listdir('mscoco/val2017')
    images.sort()
    print(images)
    json_obj = {}
    json_obj[''] = []
    detect_time_list = []
    pre_process_list = []
    post_process_list = []
    post_process_list_two = []
    main()