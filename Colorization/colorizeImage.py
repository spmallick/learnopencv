# This code is written by Sunita Nayak at BigVision LLC. It is based on the OpenCV project.
# It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example: python3 colorizeImage.py --input greyscaleImage.png

import numpy as np
import cv2 as cv
import argparse
import os.path
import time

parser = argparse.ArgumentParser(description='Colorize GreyScale Image')
parser.add_argument('--input', help='Path to image.')
parser.add_argument("--device", default="cpu", help="Device to inference on")
args = parser.parse_args()

if args.input is None:
    print('Please give the input greyscale image name.')
    print('Usage example: python3 colorizeImage.py --input greyscaleImage.png')
    exit()

if not os.path.isfile(args.input):
    print('Input file does not exist')
    exit()

print("Input image file: ", args.input)

# Read the input image
frame = cv.imread(args.input)

# Specify the paths for the 2 model files
protoFile = "./models/colorization_deploy_v2.prototxt"
weightsFile = "./models/colorization_release_v2.caffemodel"

# Load the cluster centers
pts_in_hull = np.load('./pts_in_hull.npy')

# Read the network into Memory
net = cv.dnn.readNetFromCaffe(protoFile, weightsFile)

if args.device == "cpu":
    net.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
    print("Using CPU device")
elif args.device == "gpu":
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
    print("Using GPU device")

# populate cluster centers as 1x1 convolution kernel
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

# from opencv sample
W_in = 224
H_in = 224

start = time.time()

img_rgb = (frame[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
img_l = img_lab[:, :, 0]  # pull out L channel

# resize lightness channel to network input size
img_l_rs = cv.resize(img_l, (W_in, H_in))
img_l_rs -= 50  # subtract 50 for mean-centering

net.setInput(cv.dnn.blobFromImage(img_l_rs))
ab_dec = net.forward()[0, :, :, :].transpose((1, 2, 0))  # this is our result

(H_orig, W_orig) = img_rgb.shape[:2]  # original image size
ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
img_lab_out = np.concatenate((img_l[:, :, np.newaxis],ab_dec_us), axis=2)  # concatenate with original image L
img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

end = time.time()
print("Time taken : {:0.5f} secs".format(end - start))

outputFile = args.input[:-4] + '_colorized.png'
cv.imwrite(outputFile, (img_bgr_out * 255).astype(np.uint8))
print('Colorized image saved as ' + outputFile)
print('Done !!!')

