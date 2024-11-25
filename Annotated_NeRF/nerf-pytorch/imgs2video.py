import cv2
import os
from natsort import natsorted

def create_video_from_images(image_folder, output_video_path, fps=30, frame_size=None):
    # Get list of image files in the directory
    images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    images = natsorted(images)  # Sort images naturally

    if not images:
        print("No images found in the directory.")
        return
    
    # Load the first image to determine the frame size if not specified
    first_image_path = os.path.join(image_folder, images[0])
    first_image = cv2.imread(first_image_path)

    if frame_size is None:
        frame_size = (first_image.shape[1], first_image.shape[0])  # (width, height)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change 'mp4v' to other codec if needed
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)
    
    # Iterate through images and write them to the video
    for image_file in images:
        image_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(image_path)
        
        if frame is None:
            print(f"Could not read {image_path}. Skipping...")
            continue

        # Resize frame if frame size is specified differently
        if (frame.shape[1], frame.shape[0]) != frame_size:
            frame = cv2.resize(frame, frame_size)
        
        out.write(frame)  # Write frame to video

    # Release everything when job is finished
    out.release()
    print(f"Video saved at {output_video_path}")

# Example usage
image_folder = '/home/opencvuniv/Work/somusan/3dv/data/org_pothos/nerf-pytorch/logs/moose/renderonly_path_009999'  # Replace with your directory path
output_video_path = '/home/opencvuniv/Work/somusan/3dv/data/org_pothos/nerf-pytorch/logs/moose/renderonly_path_009999/moose_10k.mp4'
create_video_from_images(image_folder, output_video_path, fps=30)
