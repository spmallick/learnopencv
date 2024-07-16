# Import the necessary libraries
import cv2
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import sys

# Define a class for publishing camera images as ROS 2 messages
class VodImagePublisher(Node):

    def __init__(self, capture):
        # node name is mentioned here, 'vod_img_pub'
        super().__init__('vod_img_pub')
        
        # Create a publisher for the '/cam/image_raw' topic with a queue size of 1
        self.publisher_ = self.create_publisher(Image, '/cam/image_raw', 1)
        
        # a timer is created with a callback to execute every 0.5 seconds
        timer_period = 0.1 
        self.timer = self.create_timer(timer_period, self.image_callback)
        
        # Initialize the image counter
        self.i = 0
        
        # Store the video capture object
        self.capture = capture

    # Callback function for capturing and publishing images
    def image_callback(self):
        # Check if the video capture is open
        if self.capture.isOpened():
            # Read a frame from the video capture
            ret, frame = self.capture.read()
            
            # Resize the frame to 640x480
            frame = cv2.resize(frame, (640, 480))
            
            # Create an Image message
            msg = Image()
            
            
            # Set the message header timestamp and frame ID
            msg.header.stamp = Node.get_clock(self).now().to_msg()
            msg.header.frame_id = "base"
            
            # Set the width and height of the image
            msg.width = frame.shape[1]
            msg.height = frame.shape[0]
            
            # Set the encoding format of the image
            msg.encoding = "bgr8"
            
            # Specify the endianness of the image data
            msg.is_bigendian = False
            
            # Set the step size (row length in bytes)
            msg.step = np.shape(frame)[2] * np.shape(frame)[1]
            
            # Convert the frame to bytes and set the image data
            msg.data = np.array(frame).tobytes()
            
            # Publish the image message
            self.publisher_.publish(msg)
            
            # Log and print the number of images published
            self.get_logger().info('%d Images Published' % self.i)

        # Increment the image counter
        self.i += 1

# Main function to initialize and run the node
def main(args=None):
    # Get the video path from command-line arguments
    video_path = sys.argv[1]
    
    # Open the video file
    # Set the buffer size for the video capture
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    
    # Initialize the rclpy library
    rclpy.init(args=args)
    
    # Initialize the CamImagePublisher object
    cam_img_publisher = None
    
    # Check if the video capture is opened successfully
    if not capture.isOpened():
        # Print an error message if the video capture cannot be opened
        print("Error: Could not open video stream from webcam.")
        
        # Destroy the node and shutdown rclpy
        cam_img_publisher.destroy_node()
        rclpy.shutdown()
        
        # Release the video capture object
        capture.release()
    else:
        # Create the VodImagePublisher object and start the node
        cam_img_publisher = VodImagePublisher(capture)
        
        # Spin the node to keep it running
        rclpy.spin(cam_img_publisher)



if __name__ == '__main__':
    # Call the main function
    main()
