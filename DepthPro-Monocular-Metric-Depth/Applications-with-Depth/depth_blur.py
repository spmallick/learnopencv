import cv2
import numpy as np

# Load RGB image and depth map
rgb_image = cv2.imread("/home/jaykumaran/Office/ml-depth-pro/Depth-Input/new/satya/0F9A9948.JPG")  # Replace with your image path
depth_map = cv2.imread("/home/jaykumaran/Office/ml-depth-pro/Depth-Input/new/satya_ops/0F9A9946_depth_grayscale.jpg", cv2.IMREAD_GRAYSCALE)  # Grayscale depth map



# Normalize depth map to range [0, 1]
depth_map_normalized = cv2.normalize(depth_map.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

# Convert normalized depth map back to uint8 (0-255 range)
depth_map_uint8 = (depth_map_normalized * 255).astype(np.uint8)

# Automatically infer focus range
min_depth = np.min(depth_map_normalized)
focus_margin = 0.22  # start with 10% margin for focus range -> 0.1
focus_near = int(min_depth * 255)
focus_far = int((min_depth + focus_margin) * 255)

# Debug: Print focus range
print(f"Focus range: {focus_near} to {focus_far}")

# Create a binary mask for the focus region
focus_mask = cv2.inRange(depth_map_uint8, focus_near, focus_far)

# Apply Gaussian blur to the entire image
blurred_image = cv2.GaussianBlur(rgb_image, (51, 51), 0)

# Convert focus mask to 3 channels for blending
focus_mask_color = cv2.merge([focus_mask, focus_mask, focus_mask])

# Blend images: Keep original where mask is white, blur otherwise
result = np.where(focus_mask_color == 255, rgb_image, blurred_image)

# Save and display the result
cv2.imshow("Depth Blur Effect", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("satya_output_depth_blur_standing.jpg", result)