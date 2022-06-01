import cv2
import mediapipe as mp
import numpy as np


def background_blur(image):
    """
    Args:
        image: frame captured by camera

    Returns:
        The image with a blurred background
    """
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=1) as selfie_segmentation:
        # Convert Image to RGB from BGR
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # Make image readable before processing to increase the performance
        image.flags.writeable = False

        results = selfie_segmentation.process(image)
        image.flags.writeable = True
        # Convert Image to BGR from RGB
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create a condition for blurring the background
        condition = np.stack(
            (results.segmentation_mask,) * 3, axis=-1) > 0.1

        # Create a blurred background image
        bg_image = cv2.GaussianBlur(image, (55, 55), 0)
        # Remove map the image on blurred background
        output_image = np.where(condition, image, bg_image)

        # Flip the output
        output_image = cv2.flip(output_image, 1)
    return output_image
