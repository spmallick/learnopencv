from PIL import Image
import src.depth_pro as depth_pro
import numpy as np
import matplotlib.pyplot as plt
import tempfile
import os
import cv2
import torch

import warnings
warnings.filterwarnings("ignore", "You are using `torch.load` with `weights_only=False`*.")

# Set the device
# device = 'cpu'
device = 'cuda:0'

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms(device = 'cuda:0',
                                                         precision=torch.half)
model.eval()

def resize_image(image_path, max_size=1536):
    with Image.open(image_path) as img:
        # Calculate the new size while maintaining aspect ratio
        ratio = max_size / max(img.size)
        new_size = tuple([int(x * ratio) for x in img.size])

        # Resize the image
        img = img.resize(new_size, Image.LANCZOS)

        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_file:
            img.save(temp_file, format="PNG")
            return temp_file.name

def show_distance_in_opencv(inverse_depth, focal_length_px, depth):
    """Displays the OpenCV window with constant distance display."""
    # Clip and normalize the inverse depth for visualization
    inverse_depth = np.clip(inverse_depth, 1e-6, 10)  # Avoid divide-by-zero
    normalized_image = cv2.normalize(inverse_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    color_map = cv2.applyColorMap(normalized_image, cv2.COLORMAP_INFERNO)

    # Initialize mouse position
    mouse_position = [0, 0]

    def update_mouse_position(event, x, y, flags, param):
        """Update the mouse position on mouse events."""
        if event == cv2.EVENT_MOUSEMOVE or event == cv2.EVENT_LBUTTONDOWN:
            mouse_position[0], mouse_position[1] = x, y

    cv2.namedWindow("Interactive Depth Viewer")
    cv2.setMouseCallback("Interactive Depth Viewer", update_mouse_position)
    


    while True:
        display_image = color_map.copy()

        # Get the current mouse position
        x, y = mouse_position
    
        cv2.circle(display_image, (x,y), radius=5, color=(255,255,255), thickness=-1)
        # Ensure the position is within the image bounds
        if 0 <= x < inverse_depth.shape[1] and 0 <= y < inverse_depth.shape[0]:
            inv_depth = inverse_depth[y, x]
            z = 1.0 / inv_depth if inv_depth > 1e-6 else float('inf')
            cv2.putText(display_image, f"Distance: {z:.2f} m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

  
        
        # Display the updated image
        cv2.imshow("Interactive Depth Viewer", display_image)

        # Exit if 'Esc' is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

def predict_depth(input_image):
    temp_file = None
    try:
        # Resize the input image
        temp_file = resize_image(input_image)

        # Preprocess the image
        result = depth_pro.load_rgb(temp_file)
        image = result[0]
        f_px = result[-1]  # Assuming f_px is the last item in the returned tuple
        image = transform(image)
        image = image.to(device)

        # Run inference
        prediction = model.infer(image, f_px=f_px)
        depth = prediction["depth"]  # Depth in [m]
        focal_length_px = prediction["focallength_px"]  # Focal length in pixels
        
        print("Focal length: ", focal_length_px)

        # Convert depth to numpy array if it's a torch tensor
        if isinstance(depth, torch.Tensor):
            depth = depth.cpu().numpy()

        # Ensure depth is a 2D numpy array
        if depth.ndim != 2:
            depth = depth.squeeze()

        # Calculate inverse depth
        inverse_depth = 1.0 / depth
    

        # Display the interactive OpenCV window
        show_distance_in_opencv(inverse_depth, focal_length_px, depth)

        # Create a color map for saving
        plt.figure(figsize=(15.36, 15.36), dpi=100)
        plt.imshow(inverse_depth, cmap='inferno')
        plt.colorbar(label='Inverse Depth')
        plt.title('Predicted Inverse Depth Map')
        plt.axis('off')

        # Save the plot to a file
        output_path = "satya_visualize.png"
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()

        return output_path, f"Focal length: {focal_length_px:.2f} pixels"
    except Exception as e:
        return None, f"An error occurred: {str(e)}"
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)


input_image_path = "/home/jaykumaran/Downloads/PXL_20241219_093842755.MP.jpg"  
output_path, message = predict_depth(input_image_path)
print(f"Output saved to: {output_path}")
print(message)
