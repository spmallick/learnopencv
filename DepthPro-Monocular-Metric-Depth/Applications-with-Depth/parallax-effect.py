import cv2
import numpy as np

# Load the image and depth map
image = cv2.imread('/home/jaykumaran/Office/ml-depth-pro/Depth-Input/PXL_20250108_101012224.jpg', cv2.IMREAD_COLOR)
depth_map = cv2.imread('/home/jaykumaran/Office/ml-depth-pro/demo_output/new/PXL_20250108_101012224normal_depth_grayscale.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure image and depth map are loaded
if image is None:
    print("Error: Unable to load image.")
    exit()
if depth_map is None:
    print("Error: Unable to load depth map.")
    exit()

# Normalize the depth map (convert to range 0-1)
depth_map = cv2.normalize(depth_map.astype('float32'), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

# Debug: Save normalized depth map

# Parameters for parallax effect
frame_count = 300  # Number of frames for 5 seconds at 30 FPS
displacement = 25 # Maximum pixel shift in any direction
fps = 60  # Frames per second

# Create a video writer for MP4
h, w, _ = image.shape
out = cv2.VideoWriter('parallax_effect.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Create a named window to avoid errors
cv2.namedWindow('Parallax Effect', cv2.WINDOW_NORMAL)

# Generate frames
for t in range(frame_count):
    # Calculate x and y displacement to create a smooth looping effect
    dx = displacement * np.sin(2 * np.pi * t / frame_count)
    dy = displacement * np.cos(2 * np.pi * t / frame_count)

    # Warp the image based on the depth map and displacement
    flow_x = depth_map * dx
    flow_y = depth_map * dy

    # Debug: Check flow ranges
    if t == 0:
        print(f"Flow X Range: {flow_x.min()} to {flow_x.max()}")
        print(f"Flow Y Range: {flow_y.min()} to {flow_y.max()}")

    # Create a meshgrid for remapping
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + flow_x).astype(np.float32)
    map_y = (y + flow_y).astype(np.float32)

    # Remap the image
    warped_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # Debug: Ensure warped image is valid
    if t == 0:
        print(f"Warped Image Shape: {warped_image.shape}")
        print(f"Warped Image Dtype: {warped_image.dtype}")

    # Write the frame to the video
    out.write(warped_image)

    # Show the effect frame by frame
    cv2.imshow('Parallax Effect', warped_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
out.release()
cv2.destroyAllWindows()

