import cv2


def fetch_zoom_factor(distance):
    """
    Args:
        distance: the default distance between two fingers

    Returns:
        The factor by which the image should be zoomed into
    """

    finger_range = (300 - 150)  # Output max - input min
    zoom_range = (2 - 1)  # Input max - input min
    addval = distance - 150 > 0  # check if the distance is positive
    zoom_fact = (addval and (((distance - 150) * zoom_range) / finger_range)) + 1
    return zoom_fact


def zoom_center(image, zoom_fact):
    """
    Args:
        image: current frame
        zoom_fact: Current zoom factor

    Returns:
        zoomed image with center focus
    """
    # Get Image coordinates
    y_size = image.shape[0]
    x_size = image.shape[1]

    # define new boundaries
    x1 = int(0.5 * x_size * (1 - 1 / zoom_fact))
    x2 = int(x_size - 0.5 * x_size * (1 - 1 / zoom_fact))
    y1 = int(0.5 * y_size * (1 - 1 / zoom_fact))
    y2 = int(y_size - 0.5 * y_size * (1 - 1 / zoom_fact))

    # first crop image then scale
    img_cropped = image[y1:y2, x1:x2]
    zoomed_image = cv2.resize(img_cropped, None, fx=zoom_fact, fy=zoom_fact)
    return zoomed_image
