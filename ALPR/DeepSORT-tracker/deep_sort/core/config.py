#! /usr/bin/env python
# coding=utf-8
from easydict import EasyDict as edict
import cv2
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf

from deep_sort import nn_matching
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = "./data/vehicle-detector/voc.names"
__C.YOLO.ANCHORS              = [12,16, 19,36, 40,28, 36,75, 76,55, 72,146, 142,110, 192,243, 459,401]
__C.YOLO.ANCHORS_V3           = [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]
__C.YOLO.ANCHORS_TINY         = [23,27, 37,58, 81,82, 81,82, 135,169, 344,319]
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.STRIDES_TINY         = [16, 32]
__C.YOLO.XYSCALE              = [1.2, 1.1, 1.05]
__C.YOLO.XYSCALE_TINY         = [1.05, 1.05]
__C.YOLO.ANCHOR_PER_SCALE     = 3
__C.YOLO.IOU_LOSS_THRESH      = 0.5

# Train options
__C.TRAIN                     = edict()

__C.TRAIN.ANNOT_PATH          = "./data/dataset/val2017.txt"
__C.TRAIN.BATCH_SIZE          = 2
# __C.TRAIN.INPUT_SIZE            = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
__C.TRAIN.INPUT_SIZE          = 416
__C.TRAIN.DATA_AUG            = True
__C.TRAIN.LR_INIT             = 1e-3
__C.TRAIN.LR_END              = 1e-6
__C.TRAIN.WARMUP_EPOCHS       = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 30

# TEST options
__C.TEST                      = edict()

__C.TEST.ANNOT_PATH           = "./data/dataset/val2017.txt"
__C.TEST.BATCH_SIZE           = 2
__C.TEST.INPUT_SIZE           = 416
__C.TEST.DATA_AUG             = False
__C.TEST.DECTECTED_IMAGE_PATH = "./data/detection/"
__C.TEST.SCORE_THRESHOLD      = 0.25
__C.TEST.IOU_THRESHOLD        = 0.5

# DEEP SORT
__C.max_cosine_distance       = 0.4
__C.nn_budget                 = None
__C.nms_max_overlap           = 1.0
__C.model_filename            = './model_data/mars-small128.pb'

# GENERAL CONSTANTS
__C.live_stream_link = "https://www.youtube.com/watch?v=t-phGBfPEZ4"
__C.obj_image_link = "s3://cam.frames/object_detection/"
__C.video_name = "./data/video/test_0.mp4"

__C.encoder = gdet.create_box_encoder(cfg.model_filename, batch_size=1)
__C.metric = nn_matching.NearestNeighborDistanceMetric("cosine", cfg.max_cosine_distance, cfg.nn_budget)
__C.tracker = Tracker(cfg.metric)

__C.allowed_classes = ['car','motorbike','truck','bus']

__C.area2 = [[200, 400], [2000, 400], [2000, 800], [200, 800]]
__C.area1 = [[0, 1400], [2592, 1400], [2592, 1944], [0, 1944]]

# OBJECT DETECTION
modelConfig = './data/vehicle-detector/yolo-tiny.cfg'
modelWeights = './data/vehicle-detector/yolo-tiny.weights'
net = cv2.dnn.readNet(modelConfig, modelWeights)
__C.all_model = cv2.dnn_DetectionModel(net)
__C.csv_file_name = './csv_files/object_detection.csv'
