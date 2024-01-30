import cv2


# Function to get the four corner coordinates of a frame
def get_corner_coordinates(frame):
    height, width = frame.shape[:2]
    top_left = (0, 0)
    top_right = (width, 0)
    bottom_left = (0, height)
    bottom_right = (width, height)
    return top_left, top_right, bottom_left, bottom_right


# Path to the video file
video_path = "path/to/your/video.mp4"

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the video frame by frame
while True:
    ret, frame = cap.read()

    # If frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Get corner coordinates
    corners = get_corner_coordinates(frame)
    print("Corner coordinates:", corners)

    # You can also display the frame with corner points
    # for corner in corners:
    #     cv2.circle(frame, corner, 5, (0, 0, 255), -1)
    # cv2.imshow('Frame with Corners', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the video capture object
cap.release()
cv2.destroyAllWindows()
