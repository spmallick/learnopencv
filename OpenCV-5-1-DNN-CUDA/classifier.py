"""ImageNet classification with OpenCV preprocessing and a choice of ONNX runtimes."""

import cv2 as cv
import numpy as np


CROP = 224
# ImageNet training statistics in RGB order. blobFromImagesWithParams subtracts
# the mean and then applies the scale to each channel after swapping B and R.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def make_blob(paths, resize_short=256):
    """Decode the files, resize their short edges, and return an FP32 NCHW batch."""
    crops = []
    for path in paths:
        image = cv.imread(str(path), cv.IMREAD_COLOR)
        if image is None:
            raise ValueError(f'OpenCV could not decode {path}')
        height, width = image.shape[:2]
        short = min(height, width)
        size = (width * resize_short // short, height * resize_short // short)
        resized = cv.resize(image, size, interpolation=cv.INTER_LINEAR)
        # Center crop without stretching the original aspect ratio.
        x, y = round((size[0] - CROP) / 2), round((size[1] - CROP) / 2)
        crops.append(resized[y:y + CROP, x:x + CROP])

    params = cv.dnn.Image2BlobParams()
    params.size = (CROP, CROP)
    params.swapRB = True  # imread returns BGR. Both pretrained models expect RGB.
    # One OpenCV call computes (pixel / 255 - mean) / std and lays out N,C,H,W.
    params.mean = tuple(255 * value for value in MEAN)
    params.scalefactor = tuple(1 / (255 * value) for value in STD)
    return cv.dnn.blobFromImagesWithParams(crops, params)


def top_five(logits, labels):
    """Rank each image's scores. Softmax would leave their ordering unchanged."""
    order = cv.sortIdx(logits, cv.SORT_EVERY_ROW | cv.SORT_DESCENDING)[:, :5]
    ids = order.tolist()
    # A label file is optional so readers can try models with a different vocabulary.
    names = [labels[row[0]] for row in ids] if labels else []
    return ids, names


def opencv_runner(args):
    """Configure the net before its first forward, then accept host input/output."""
    net = cv.dnn.readNetFromONNX(str(args.model), cv.dnn.ENGINE_OPENCV)
    if args.backend == 'opencv-cuda':
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
        net.enableTF32(args.tf32 == 'on')
    else:
        net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

    def forward(blob):
        net.setInput(blob)
        return net.forward()  # 5.1 copies the result back to host before returning.

    return forward


def ort_runner(args, ort, profile=False):
    """Return the CUDA EP session as well as the same host-array forward interface."""
    cuda = 'CUDAExecutionProvider'
    if cuda not in ort.get_available_providers():
        raise RuntimeError('Install a CUDA-enabled ONNX Runtime matching your CUDA and cuDNN.')
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    if profile:
        opts.enable_profiling = True
        opts.profile_file_prefix = str(args.output / 'ort-placement')
    # Leave ORT CUDA Graphs off: this comparison uses the normal host session.run API.
    provider_options = {'device_id': '0', 'use_tf32': str(int(args.tf32 == 'on'))}
    session = ort.InferenceSession(str(args.model), opts, providers=[(cuda, provider_options)])
    # ORT can recover from provider initialization failures by constructing a CPU
    # session. Also turn off its Python wrapper's retry-on-EP-failure during run().
    if session.get_providers()[0] != cuda:
        raise RuntimeError(f'ORT failed to initialize CUDA: {session.get_providers()}')
    session.disable_fallback()
    actual_tf32 = session.get_provider_options()[cuda].get('use_tf32')
    if actual_tf32 != provider_options['use_tf32']:
        raise RuntimeError(f'Cannot confirm ORT use_tf32 setting: got {actual_tf32!r}')
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError('This classifier expects one NCHW input and one logits output.')
    name = inputs[0].name

    def forward(blob):
        # With CPU NumPy inputs/outputs, ORT transfers data and synchronizes the result.
        return session.run(None, {name: blob})[0]

    return forward, session


def check_scores(scores, batch, labels):
    if scores.ndim != 2 or scores.shape[0] != batch or scores.shape[1] < 1:
        raise ValueError(f'Expected [batch, classes], got {scores.shape} for batch {batch}')
    if labels and len(labels) != scores.shape[1]:
        raise ValueError(f'{len(labels)} labels for {scores.shape[1]} output classes')
    if not np.isfinite(scores).all():
        raise ValueError('Model returned a non-finite score')
