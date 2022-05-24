import argparse
import os
import time

import cv2


def detectFaceOpenCVDnn(net, frame, framework="caffe", conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    if framework == "caffe":
        blob = cv2.dnn.blobFromImage(
            frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False,
        )
    else:
        blob = cv2.dnn.blobFromImage(
            frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False,
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("--video", type=str, default="", help="Path to video file")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="Device to use",
    )
    parser.add_argument(
        "--framework",
        type=str,
        default="caffe",
        choices=["caffe", "tf"],
        help="Type of network to run",
    )
    args = parser.parse_args()

    framework = args.framework
    source = args.video
    device = args.device

    # OpenCV DNN supports 2 networks.
    # 1. FP16 version of the original Caffe implementation ( 5.4 MB )
    # 2. 8 bit Quantized version using TensorFlow ( 2.7 MB )

    if framework == "caffe":
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

    outputFolder = "output-dnn-videos"
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)

    if source:
        cap = cv2.VideoCapture(source)
        outputFile = os.path.basename(source)[:-4] + ".avi"
    else:
        cap = cv2.VideoCapture(0, cv2.CAP_V4L)
        outputFile = "grabbed_from_camera.avi"

    hasFrame, frame = cap.read()

    vid_writer = cv2.VideoWriter(
        os.path.join(outputFolder, outputFile),
        cv2.VideoWriter_fourcc("M", "J", "P", "G"),
        15,
        (frame.shape[1], frame.shape[0]),
    )

    frame_count = 0
    tt_opencvDnn = 0

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_count += 1
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

        cv2.imshow("Face Detection Comparison", outOpencvDnn)

        vid_writer.write(outOpencvDnn)

        if frame_count == 1:
            tt_opencvDnn = 0

        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()
    vid_writer.release()
