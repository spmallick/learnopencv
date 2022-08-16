import cv2
import mediapipe as mp
from vidstab import VidStab
from pyvirtualcam import PixelFormat
import pyvirtualcam
import platform

# global variables
gb_zoom = 1.4


def zoom_at(image, coord=None, zoom_type=None):
    """
    Args:
        image: frame captured by camera
        coord: coordinates of face(nose)
        zoom_type:Is it a transition or normal zoom
    Returns:
        Image with cropped image
    """
    global gb_zoom
    # If zoom_type is transition check if Zoom is already done else zoom by 0.1 in current frame
    if zoom_type == 'transition' and gb_zoom < 3.0:
        gb_zoom = gb_zoom + 0.1

    # If zoom_type is normal zoom check if zoom more than 1.4 if soo zoom out by 0.1 in each frame
    if gb_zoom != 1.4 and zoom_type is None:
        gb_zoom = gb_zoom - 0.1

    zoom = gb_zoom
    # If coordinates to zoom around are not specified, default to center of the frame
    cy, cx = [i / 2 for i in image.shape[:-1]] if coord is None else coord[::-1]

    # Scaling the image using getRotationMatrix2D to appropriate zoom
    rot_mat = cv2.getRotationMatrix2D((cx, cy), 0, zoom)

    # Use warpAffine to make sure that  lines remain parallel
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def frame_manipulate(img):
    """
    Args:
        image: frame captured by camera
    Returns:
        Image with manipulated output
    """
    # Mediapipe face set up
    mp_face_detection = mp.solutions.face_detection
    with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:

        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(img)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Set the default values to None
        coordinates = None
        zoom_transition = None
        if results.detections:
            for detection in results.detections:
                height, width, channels = img.shape

                # Fetch coordinates of nose, right ear and left ear
                nose = detection.location_data.relative_keypoints[2]
                right_ear = detection.location_data.relative_keypoints[4]
                left_ear = detection.location_data.relative_keypoints[5]

                #  get coordinates for right ear and left ear
                right_ear_x = int(right_ear.x * width)
                left_ear_x = int(left_ear.x * width)

                # Fetch coordinates for the nose and set as center
                center_x = int(nose.x * width)
                center_y = int(nose.y * height)
                coordinates = [center_x, center_y]

                # Check the distance between left ear and right ear if distance is less than 120 pixels zoom in
                if (left_ear_x - right_ear_x) < 120:
                    zoom_transition = 'transition'

        # Perform zoom on the image
        img = zoom_at(img, coord=coordinates, zoom_type=zoom_transition)

    return img


def main():
    # Video Stabilizer
    device_val = None
    stabilizer = VidStab()

    # For webcam input:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # set new dimensions to cam object (not cap)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 120)

    # Check OS
    os = platform.system()
    if os == "Linux":
        device_val = "/dev/video2"

    # Start virtual camera
    with pyvirtualcam.Camera(1280, 720, 120, device=device_val, fmt=PixelFormat.BGR) as cam:
        print('Virtual camera device: ' + cam.device)

        while True:
            success, img = cap.read()
            img = frame_manipulate(img)
            # Stabilize the image to make sure that the changes with Zoom are very smooth
            img = stabilizer.stabilize_frame(input_frame=img,
                                             smoothing_window=2, border_size=-20)
            # Resize the image to make sure it does not crash pyvirtualcam
            img = cv2.resize(img, (1280, 720),
                             interpolation=cv2.INTER_CUBIC)

            cam.send(img)
            cam.sleep_until_next_frame()


if __name__ == '__main__':
    main()
