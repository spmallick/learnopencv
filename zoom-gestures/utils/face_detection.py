import cv2
import mediapipe as mp


def face_detect(image):
    """
    Args:
        image: frame captured by camera

    Returns:
        The number of faces
    """
    # Use Mediapipe face detection
    mp_face_detection = mp.solutions.face_detection

    # choose facedetection criteria
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        # Make the image non-writeable since the detection needs no write access
        # Doing so also improves performance
        image.flags.writeable = False
        # Convert image from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # process the image
        results = face_detection.process(image)
        # If any face is detected return the number of faces
        if results.detections:
            return len(results.detections)
