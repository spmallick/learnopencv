import argparse
import os
import time

import cv2


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face detection")
    parser.add_argument("--video", type=str, default="", help="Path to video file")
    args = parser.parse_args()

    source = args.video

    faceCascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")

    outputFolder = "output-haar-videos"
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
        25,
        (frame.shape[1], frame.shape[0]),
    )

    frame_count = 0
    tt_opencvHaar = 0

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

        cv2.imshow("Face Detection Comparison", outOpencvHaar)

        vid_writer.write(outOpencvHaar)
        if frame_count == 1:
            tt_opencvHaar = 0

        key = cv2.waitKey(5)
        if key == 27:
            break

    cv2.destroyAllWindows()
    vid_writer.release()
