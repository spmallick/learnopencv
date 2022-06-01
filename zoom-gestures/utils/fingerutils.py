WRIST = 0
INDEX_FINGER_PIP = 6
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20


def is_fist_closed(points):
    """
    Args:
        points: landmarks from mediapipe

    Returns:
        boolean check if fist is closed
    """

    return points[MIDDLE_FINGER_MCP].y < points[MIDDLE_FINGER_TIP].y and points[PINKY_MCP].y < points[PINKY_TIP].y and \
        points[RING_FINGER_MCP].y < points[RING_FINGER_TIP].y


def hand_down(points):
    """
    Args:
        points: landmarks from mediapipe

    Returns:
        boolean check if hand is down i.e. inverted
    """

    return points[MIDDLE_FINGER_TIP].y > points[WRIST].y


def hand_up(points):
    """
    Args:
        points: landmarks from mediapipe

    Returns:
        boolean check if hand is up
    """
    return points[MIDDLE_FINGER_TIP].y < points[WRIST].y


def two_signal(points):
    """
    Args:
        points: landmarks from mediapipe

    Returns:
        boolean check if fingers show two
    """

    return points[INDEX_FINGER_TIP].y < points[INDEX_FINGER_PIP].y and points[MIDDLE_FINGER_TIP].y < points[
        MIDDLE_FINGER_PIP].y and points[RING_FINGER_PIP].y < points[
        RING_FINGER_TIP].y and \
        points[PINKY_PIP].y < \
        points[PINKY_TIP].y


def three_signal(points):
    """
    Args:
        points: landmarks from mediapipe

    Returns:
        boolean check if fingers show three
    """

    return points[INDEX_FINGER_TIP].y < points[INDEX_FINGER_PIP].y and points[MIDDLE_FINGER_TIP].y < points[
        MIDDLE_FINGER_PIP].y and points[RING_FINGER_PIP].y > points[
        RING_FINGER_TIP].y and \
        points[PINKY_PIP].y < \
        points[PINKY_TIP].y
