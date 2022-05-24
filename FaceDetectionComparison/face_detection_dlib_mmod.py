import argparse
import os
import time

import cv2
import dlib


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

    print(frameWidth, frameHeight, inWidth, inHeight)
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
    parser.add_argument("--video", type=str, default="", help="Path to video file")
    args = parser.parse_args()

    source = args.video

    mmodFaceDetector = dlib.cnn_face_detection_model_v1(
        "models/mmod_human_face_detector.dat",
    )

    outputFolder = "output-mmod-videos"
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
    tt_dlibMmod = 0

    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            break

        frame_count += 1
        t = time.time()

        outDlibMMOD, bboxes = detectFaceDlibMMOD(mmodFaceDetector, frame)
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

        cv2.imshow("Face Detection Comparison", outDlibMMOD)

        vid_writer.write(outDlibMMOD)

        if frame_count == 1:
            tt_dlibMmod = 0

        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()
    vid_writer.release()
