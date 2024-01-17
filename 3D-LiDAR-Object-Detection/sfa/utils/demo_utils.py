"""
'''
///////////////////////////////////////
3D LiDAR Object Detection - ADAS
Pranav Durai
//////////////////////////////////////
'''
# Description: Demonstration utils script
"""

import argparse
import sys
import os
import warnings
import zipfile

warnings.filterwarnings("ignore", category=UserWarning)

from easydict import EasyDict as edict
import numpy as np
import wget
import torch
import cv2

src_dir = os.path.dirname(os.path.realpath(__file__))
while not src_dir.endswith("sfa"):
    src_dir = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

from utils.misc import make_folder, time_synchronized
from utils.evaluation_utils import decode, post_processing
from utils.torch_utils import _sigmoid


def parse_demo_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for the implementation')
    parser.add_argument('--saved_fn', type=str, default='fpn_resnet_18', metavar='FN',
                        help='The name using for saving logs, models,...')
    parser.add_argument('-a', '--arch', type=str, default='fpn_resnet_18', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_path', type=str,
                        default='../checkpoints/fpn_resnet_18/fpn_resnet_18_epoch_300.pth', metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--foldername', type=str, default='2011_09_26_drive_0014_sync', metavar='FN',
                        help='Folder name for demostration dataset')
    parser.add_argument('--K', type=int, default=50,
                        help='the number of top K')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='GPU index to use.')
    parser.add_argument('--peak_thresh', type=float, default=0.2)
    parser.add_argument('--output_format', type=str, default='image', metavar='PATH',
                        help='the type of the test output (support image or video)')
    parser.add_argument('--output-width', type=int, default=608,
                        help='the width of showing output, the height maybe vary')

    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True
    configs.distributed = False  # For testing on 1 GPU only

    configs.input_size = (608, 608)
    configs.hm_size = (152, 152)
    configs.down_ratio = 4
    configs.max_objects = 50

    configs.imagenet_pretrained = False
    configs.head_conv = 64
    configs.num_classes = 3
    configs.num_center_offset = 2
    configs.num_z = 1
    configs.num_dim = 3
    configs.num_direction = 2  # sin, cos

    configs.heads = {
        'hm_cen': configs.num_classes,
        'cen_offset': configs.num_center_offset,
        'direction': configs.num_direction,
        'z_coor': configs.num_z,
        'dim': configs.num_dim
    }

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti', 'demo')
    configs.calib_path = os.path.join(configs.root_dir, 'dataset', 'kitti', 'demo', 'calib.txt')
    configs.results_dir = os.path.join(configs.root_dir, 'results', configs.saved_fn)
    make_folder(configs.results_dir)

    return configs


def download_and_unzip(demo_dataset_dir, download_url):
    filename = download_url.split('/')[-1]
    filepath = os.path.join(demo_dataset_dir, filename)
    if os.path.isfile(filepath):
        print('The dataset have been downloaded')
        return
    print('\nDownloading data for demonstration...')
    wget.download(download_url, filepath)
    print('\nUnzipping the downloaded data...')
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(os.path.join(demo_dataset_dir, filename[:-4]))


def do_detect(configs, model, bevmap, is_front):
    if not is_front:
        bevmap = torch.flip(bevmap, [1, 2])

    input_bev_maps = bevmap.unsqueeze(0).to(configs.device, non_blocking=True).float()
    t1 = time_synchronized()
    outputs = model(input_bev_maps)
    outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
    outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
    # detections size (batch_size, K, 10)
    detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                        outputs['dim'], K=configs.K)
    detections = detections.cpu().numpy().astype(np.float32)
    detections = post_processing(detections, configs.num_classes, configs.down_ratio, configs.peak_thresh)
    t2 = time_synchronized()
    # Inference speed
    fps = 1 / (t2 - t1)

    return detections[0], bevmap, fps


def write_credit(img, org_author=(500, 400), text_author='github.com/maudzung', org_fps=(50, 1000), fps=None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (255, 255, 255)
    thickness = 2

    cv2.putText(img, text_author, org_author, font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.putText(img, 'Speed: {:.1f} FPS'.format(fps), org_fps, font, fontScale, color, thickness, cv2.LINE_AA)