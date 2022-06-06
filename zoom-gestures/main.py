# Imports
import cv2
import pyvirtualcam
from pyvirtualcam import PixelFormat
from utils import handsutils
import platform


def main():
    # Start video capture and set defaults
    device_val = None
    cap = cv2.VideoCapture(0)
    pref_width = 1280
    pref_height = 720
    pref_fps = 30

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, pref_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, pref_height)
    cap.set(cv2.CAP_PROP_FPS, pref_fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    os = platform.system()
    if os == "Linux":
        device_val = "/dev/video2"

    with pyvirtualcam.Camera(width, height, fps, device=device_val, fmt=PixelFormat.BGR) as cam:
        print('Virtual camera device: ' + cam.device)
        while True:
            success, img = cap.read()
            cropped_img = img[0:720, 0:400]
            img = handsutils.mediapipe_gestures(img, cropped_img)
            img = cv2.resize(img, (1280, 720))
            cam.send(img)
            cam.sleep_until_next_frame()


if __name__ == '__main__':
    """
    Main Function
    """
    main()
