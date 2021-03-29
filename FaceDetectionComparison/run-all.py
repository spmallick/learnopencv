import argparse
import os
import time

import cv2
import dlib
import numpy as np

# Model files
# OpenCV HAAR
faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

# DLIB HOG
hogFaceDetector = dlib.get_frontal_face_detector()

# DLIB MMOD
dnnFaceDetector = dlib.cnn_face_detection_model_v1(
    "models/mmod_human_face_detector.dat",
)


def detectFaceOpenCVHaar(faceCascade, frame, inHeight=300, inWidth=0):
    frameOpenCVHaar = frame.copy()
    frameHeight = frameOpenCVHaar.shape[0]
    frameWidth = frameOpenCVHaar.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameOpenCVHaarSmall = cv2.resize(frameOpenCVHaar, (inWidth, inHeight))
    frameGray = cv2.cvtColor(frameOpenCVHaarSmall, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(frameGray)
    bboxes = []
    for (x, y, w, h) in faces:
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cvRect = [
            int(x1 * scaleWidth),
            int(y1 * scaleHeight),
            int(x2 * scaleWidth),
            int(y2 * scaleHeight),
        ]
        bboxes.append(cvRect)
        cv2.rectangle(
            frameOpenCVHaar,
            (cvRect[0], cvRect[1]),
            (cvRect[2], cvRect[3]),
            (0, 255, 0),
            int(round(frameHeight / 150)),
            4,
        )
    return frameOpenCVHaar, bboxes


def detectFaceOpenCVDnn(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False,
    )

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frameHeight / 150)),
                8,
            )
    return frameOpencvDnn, bboxes


def detectFaceDlibHog(detector, frame, inHeight=300, inWidth=0):
    frameDlibHog = frame.copy()
    frameHeight = frameDlibHog.shape[0]
    frameWidth = frameDlibHog.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibHogSmall = cv2.resize(frameDlibHog, (inWidth, inHeight))

    frameDlibHogSmall = cv2.cvtColor(frameDlibHogSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibHogSmall, 0)
    bboxes = []
    for faceRect in faceRects:

        cvRect = [
            int(faceRect.left() * scaleWidth),
            int(faceRect.top() * scaleHeight),
            int(faceRect.right() * scaleWidth),
            int(faceRect.bottom() * scaleHeight),
        ]
        bboxes.append(cvRect)
        cv2.rectangle(
            frameDlibHog,
            (cvRect[0], cvRect[1]),
            (cvRect[2], cvRect[3]),
            (0, 255, 0),
            int(round(frameHeight / 150)),
            4,
        )
    return frameDlibHog, bboxes


def detectFaceDlibMMOD(detector, frame, inHeight=300, inWidth=0):
    frameDlibMMOD = frame.copy()
    frameHeight = frameDlibMMOD.shape[0]
    frameWidth = frameDlibMMOD.shape[1]
    if not inWidth:
        inWidth = int((frameWidth / frameHeight) * inHeight)

    scaleHeight = frameHeight / inHeight
    scaleWidth = frameWidth / inWidth

    frameDlibMMODSmall = cv2.resize(frameDlibMMOD, (inWidth, inHeight))

    frameDlibMMODSmall = cv2.cvtColor(frameDlibMMODSmall, cv2.COLOR_BGR2RGB)
    faceRects = detector(frameDlibMMODSmall, 0)

    bboxes = []
    for faceRect in faceRects:
        cvRect = [
            int(faceRect.rect.left() * scaleWidth),
            int(faceRect.rect.top() * scaleHeight),
            int(faceRect.rect.right() * scaleWidth),
            int(faceRect.rect.bottom() * scaleHeight),
        ]
        bboxes.append(cvRect)
        cv2.rectangle(
            frameDlibMMOD,
            (cvRect[0], cvRect[1]),
            (cvRect[2], cvRect[3]),
            (0, 255, 0),
            int(round(frameHeight / 150)),
            4,
        )
    return frameDlibMMOD, bboxes


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument(
        "--device",
        type=str,
        default="gpu",
        choices=["cpu", "gpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="caffe",
        choices=["caffe", "tf"],
        help="Type of network to run",
    )
    args = parser.parse_args()

    net_type = args.net_type
    source = args.video
    device = args.device

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original Caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using TensorFlow ( 2.7 MB )

    if net_type == "caffe":
        modelFile = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        configFile = "models/deploy.prototxt"
        net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    else:
        modelFile = "models/opencv_face_detector_uint8.pb"
        configFile = "models/opencv_face_detector.pbtxt"
        net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

    if device == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if source:
        cap = cv2.VideoCapture(source)
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L)

    hasFrame, frame = cap.read()

    outputFolder = "output-dnn-videos"
    if source:
        outputFile = os.path.basename(source)[:-4] + ".avi"
    else:
        outputFile = "grabbed_from_camera.avi"

    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    vid_writer = cv2.VideoWriter(
        os.path.join(outputFolder, outputFile),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        25,
        (frame.shape[1], frame.shape[0]),
    )

    frame_count = 0
    tt_opencvHaar = 0
    tt_opencvDnn = 0
    tt_dlibHog = 0
    tt_dlibMmod = 0

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_count += 1

        t = time.time()
        outOpencvHaar, bboxes = detectFaceOpenCVHaar(faceCascade, frame)
        tt_opencvHaar += time.time() - t
        fpsOpencvHaar = frame_count / tt_opencvHaar

        label = "OpenCV Haar; FPS : {:.2f}".format(fpsOpencvHaar)
        cv2.putText(
            outOpencvHaar,
            label,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        t = time.time()
        outOpencvDnn, bboxes = detectFaceOpenCVDnn(net, frame)
        tt_opencvDnn += time.time() - t
        fpsOpencvDnn = frame_count / tt_opencvDnn

        label = "OpenCV DNN {} FPS : {:.2f}".format(device.upper(), fpsOpencvDnn)
        cv2.putText(
            outOpencvDnn,
            label,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        t = time.time()
        outDlibHog, bboxes = detectFaceDlibHog(hogFaceDetector, frame)
        tt_dlibHog += time.time() - t
        fpsDlibHog = frame_count / tt_dlibHog

        label = "DLIB HoG; FPS : {:.2f}".format(fpsDlibHog)
        cv2.putText(
            outDlibHog,
            label,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        t = time.time()
        outDlibMMOD, bboxes = detectFaceDlibMMOD(dnnFaceDetector, frame)
        tt_dlibMmod += time.time() - t
        fpsDlibMmod = frame_count / tt_dlibMmod

        label = "DLIB MMOD; FPS : {:.2f}".format(fpsDlibMmod)
        cv2.putText(
            outDlibMMOD,
            label,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.3,
            (0, 0, 255),
            3,
            cv2.LINE_AA,
        )

        top = np.hstack([outOpencvHaar, outOpencvDnn])
        bottom = np.hstack([outDlibHog, outDlibMMOD])
        combined = np.vstack([top, bottom])
        cv2.imshow("Face Detection Comparison", combined)

        if frame_count == 1:
            tt_opencvHaar = 0
            tt_opencvDnn = 0
            tt_dlibHog = 0
            tt_dlibMmod = 0

        vid_writer.write(combined)

        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()
    vid_writer.release()
