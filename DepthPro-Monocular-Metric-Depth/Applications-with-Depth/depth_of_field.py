import cv2
import numpy as np

# Load RGB image and depth map
rgb_image = cv2.imread("/home/jaykumaran/Office/ml-depth-pro/Playful-Leopard-Cub.jpeg")
depth_map = cv2.imread("/home/jaykumaran/Office/ml-depth-pro/Playful-Leopard-Cub_inverse_depth_bnw.jpg", cv2.IMREAD_GRAYSCALE)



# Normalize depth map to range [0, 1]
depth_map_normalized = cv2.normalize(depth_map.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

# Get screen size and resize image to fit
screen_width, screen_height = 1920, 1080  # Replace with your screen resolution if needed
scale = min(screen_width / rgb_image.shape[1], screen_height / rgb_image.shape[0])
new_width = int(rgb_image.shape[1] * scale)
new_height = int(rgb_image.shape[0] * scale)

rgb_image = cv2.resize(rgb_image, (new_width, new_height))
depth_map_normalized = cv2.resize(depth_map_normalized, (new_width, new_height))

# Function to apply depth of field effect
def apply_dof(focal_depth):
    focal_range = 0.1  # Range around focal depth to remain sharp

    # Create smooth focus weights
    sharpness_weights = np.exp(-((depth_map_normalized - focal_depth) ** 2) / (2 * focal_range ** 2))
    sharpness_weights = sharpness_weights.astype(np.float32)

    # Apply Gaussian blur to the background
    blurred_image = cv2.GaussianBlur(rgb_image, (51, 51), 0)

    # Blend the original image and blurred image using sharpness weights
    sharpness_weights_3d = np.expand_dims(sharpness_weights, axis=2)  # Add a channel for blending
    dof_image = sharpness_weights_3d * rgb_image + (1 - sharpness_weights_3d) * blurred_image
    dof_image = np.clip(dof_image, 0, 255).astype(np.uint8)

    return dof_image

# Callback function for the trackbar
def on_trackbar(val):
    # Convert slider value (0-100) to focal depth (0.0-1.0)
    focal_depth = val / 100.0
    dof_image = apply_dof(focal_depth)
    cv2.imshow("Depth of Field Effect", dof_image)

# Create a window and resize it to fit the screen
cv2.namedWindow("Depth of Field Effect", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth of Field Effect", new_width, new_height)

# Create a trackbar (slider) at the top of the window
cv2.createTrackbar("Focal Plane", "Depth of Field Effect", 50, 100, on_trackbar)  # Default at middle (50)

# Show initial DOF effect
initial_dof_image = apply_dof(0.5)  # Start with focal depth at 0.5
cv2.imshow("Depth of Field Effect", initial_dof_image)

# Wait until user closes the window
cv2.waitKey(0)
cv2.destroyAllWindows()
