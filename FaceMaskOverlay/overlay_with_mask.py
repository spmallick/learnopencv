import argparse
import csv
import os
import pprint
from collections import OrderedDict

import cv2
import numpy as np
import torch

import lib.models as models
from lib.config import (
    config,
    update_config,
)
from lib.core.evaluation import decode_preds
from lib.utils import utils
from lib.utils.transforms import crop


def parse_args():

    parser = argparse.ArgumentParser(description="Face Mask Overlay")

    parser.add_argument(
        "--cfg", help="experiment configuration filename", required=True, type=str,
    )
    parser.add_argument(
        "--landmark_model",
        help="path to model for landmarks exctraction",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--detector_model",
        help="path to detector model",
        type=str,
        default="detection/face_detector.prototxt",
    )
    parser.add_argument(
        "--detector_weights",
        help="path to detector weights",
        type=str,
        default="detection/face_detector.caffemodel",
    )
    parser.add_argument(
        "--mask_image", help="path to a .png file with a mask", required=True, type=str,
    )
    parser.add_argument("--device", default="cpu", help="Device to inference on")

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    # parsing script arguments
    args = parse_args()
    device = torch.device(args.device)

    # initialize logger
    logger, final_output_dir, tb_log_dir = utils.create_logger(config, args.cfg, "demo")

    # log arguments and config values
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # init landmark model
    model = models.get_face_alignment_net(config)

    # get input size from the config
    input_size = config.MODEL.IMAGE_SIZE

    # load model
    state_dict = torch.load(args.landmark_model, map_location=device)

    # remove `module.` prefix from the pre-trained weights
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key[7:]
        new_state_dict[name] = value

    # load weights without the prefix
    model.load_state_dict(new_state_dict)
    # run model on device
    model = model.to(device)

    # init mean and std values for the landmark model's input
    mean = config.MODEL.MEAN
    mean = np.array(mean, dtype=np.float32)
    std = config.MODEL.STD
    std = np.array(std, dtype=np.float32)

    # defining prototxt and caffemodel paths
    detector_model = args.detector_model
    detector_weights = args.detector_weights

    # load model
    detector = cv2.dnn.readNetFromCaffe(detector_model, detector_weights)
    capture = cv2.VideoCapture(0)

    frame_num = 0
    while True:
        # capture frame-by-frame
        success, frame = capture.read()

        # break if no frame
        if not success:
            break

        frame_num += 1
        print("frame_num: ", frame_num)
        landmarks_img = frame.copy()
        result = frame.copy()
        result = result.astype(np.float32) / 255.0

        # get frame's height and width
        height, width = frame.shape[:2]  # 640x480

        # resize and subtract BGR mean values, since Caffe uses BGR images for input
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0),
        )
        # passing blob through the network to detect faces
        detector.setInput(blob)
        # detector output format:
        # [image_id, class, confidence, left, bottom, right, top]
        face_detections = detector.forward()

        # loop over the detections
        for i in range(0, face_detections.shape[2]):
            # extract confidence
            confidence = face_detections[0, 0, i, 2]

            # filter detections by confidence greater than the minimum threshold
            if confidence > 0.5:
                # get coordinates of the bounding box
                box = face_detections[0, 0, i, 3:7] * np.array(
                    [width, height, width, height],
                )
                (x1, y1, x2, y2) = box.astype("int")

                # show original image
                cv2.imshow("original image", frame)

                # crop to detection and resize
                resized = crop(
                    frame,
                    torch.Tensor([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2]),
                    1.5,
                    tuple(input_size),
                )

                # convert from BGR to RGB since HRNet expects RGB format
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img = resized.astype(np.float32) / 255.0
                # normalize landmark net input
                normalized_img = (img - mean) / std

                # predict face landmarks
                model = model.eval()
                with torch.no_grad():
                    input = torch.Tensor(normalized_img.transpose([2, 0, 1]))
                    input = input.to(device)
                    output = model(input.unsqueeze(0))
                    score_map = output.data.cpu()
                    preds = decode_preds(
                        score_map,
                        [torch.Tensor([x1 + (x2 - x1) / 2, y1 + (y2 - y1) / 2])],
                        [1.5],
                        score_map.shape[2:4],
                    )

                    preds = preds.squeeze(0)
                    landmarks = preds.data.cpu().detach().numpy()
                    # draw landmarks
                    for k, landmark in enumerate(landmarks, 1):
                        landmarks_img = cv2.circle(
                            landmarks_img,
                            center=(landmark[0], landmark[1]),
                            radius=3,
                            color=(0, 0, 255),
                            thickness=-1,
                        )
                        # draw landmarks' labels
                        landmarks_img = cv2.putText(
                            img=landmarks_img,
                            text=str(k),
                            org=(int(landmark[0]) + 5, int(landmark[1]) + 5),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(0, 0, 255),
                        )

                # show results by drawing predicted landmarks and their labels
                cv2.imshow("image with landmarks", landmarks_img)

                # get chosen landmarks 2-16, 30 as destination points
                # note that landmarks numbering starts from 0
                dst_pts = np.array(
                    [
                        landmarks[1],
                        landmarks[2],
                        landmarks[3],
                        landmarks[4],
                        landmarks[5],
                        landmarks[6],
                        landmarks[7],
                        landmarks[8],
                        landmarks[9],
                        landmarks[10],
                        landmarks[11],
                        landmarks[12],
                        landmarks[13],
                        landmarks[14],
                        landmarks[15],
                        landmarks[29],
                    ],
                    dtype="float32",
                )

                # load mask annotations from csv file to source points
                mask_annotation = os.path.splitext(os.path.basename(args.mask_image))[0]
                mask_annotation = os.path.join(
                    os.path.dirname(args.mask_image), mask_annotation + ".csv",
                )

                with open(mask_annotation) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=",")
                    src_pts = []
                    for i, row in enumerate(csv_reader):
                        # skip head or empty line if it's there
                        try:
                            src_pts.append(np.array([float(row[1]), float(row[2])]))
                        except ValueError:
                            continue
                src_pts = np.array(src_pts, dtype="float32")

                # overlay with a mask only if all landmarks have positive coordinates:
                if (landmarks > 0).all():
                    # load mask image
                    mask_img = cv2.imread(args.mask_image, cv2.IMREAD_UNCHANGED)
                    mask_img = mask_img.astype(np.float32)
                    mask_img = mask_img / 255.0

                    # get the perspective transformation matrix
                    M, _ = cv2.findHomography(src_pts, dst_pts)

                    # transformed masked image
                    transformed_mask = cv2.warpPerspective(
                        mask_img,
                        M,
                        (result.shape[1], result.shape[0]),
                        None,
                        cv2.INTER_LINEAR,
                        cv2.BORDER_CONSTANT,
                    )

                    # mask overlay
                    alpha_mask = transformed_mask[:, :, 3]
                    alpha_image = 1.0 - alpha_mask

                    for c in range(0, 3):
                        result[:, :, c] = (
                            alpha_mask * transformed_mask[:, :, c]
                            + alpha_image * result[:, :, c]
                        )

        # display the resulting frame
        cv2.imshow("image with mask overlay", result)

        # waiting for the escape button to exit
        k = cv2.waitKey(1)
        if k == 27:
            break

    # when everything done, release the capture
    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
