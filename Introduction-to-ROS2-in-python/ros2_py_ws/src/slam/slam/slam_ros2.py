import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from slam.pyslam import process_frame  # Assuming process_frame is a function from the SLAM package
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped, Point
import tf_transformations
import tf2_ros


class SlamPublisher(Node):

    def __init__(self):
        super().__init__('slam_pub')
        # Publisher for point cloud data with a larger queue size to handle high message rates
        self.pcd_publisher = self.create_publisher(PointCloud2, '/cam/map_points', 100)  
        # Publisher for markers representing camera poses with a moderate queue size
        self.marker_publisher = self.create_publisher(MarkerArray, '/cam/camera_pose', 10)
        # Publisher for visualizing feature extraction results with a moderate queue size
        self.feature_publisher = self.create_publisher(Image, '/cam/feature_ext_viz', 10)
        # Subscriber for raw camera images with a moderate queue size
        self.subscription = self.create_subscription(
            Image,
            '/cam/image_raw',
            self.listener_callback,
            10)  
        self.subscription  # Prevent unused variable warning
        self.bridge = CvBridge()  # Bridge to convert between ROS and OpenCV images
        self.timer_period = 0.5  # Timer period in seconds
        self.timer = self.create_timer(self.timer_period, self.publish_callback)  # Create a timer to call the publish callback periodically
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.latest_frame = None  # Variable to store the latest frame
        self.itr = 0  # Iterator counter

    def listener_callback(self, msg):
        # Callback function to handle incoming images
        self.latest_frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def broadcast_transformations(self, mapp):
        frame = mapp.frames[0]
        translation = frame.pose[0:3, 3]
        quaternion = tf_transformations.quaternion_from_matrix(frame.pose)
        self.broadcast_transformation(translation, quaternion)

    def broadcast_transformation(self, translation, quaternion):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'  # Reference frame (static)
        t.child_frame_id = 'base'  # Moving frame (dynamic)
        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]
        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]
        self.tf_broadcaster.sendTransform(t)


    def publish_callback(self):
        if self.latest_frame is not None:
            frame = self.latest_frame
            result = process_frame(frame)  # Process the frame using the SLAM library

            if result is not None:
                img, mapp = result  # Unpack the result from process_frame

                marker_array = MarkerArray()

                # Create markers for each frame in the map
                for i, frame in enumerate(mapp.frames):
                    pose = frame.pose
                    translation = pose[0:3, 3]
                    marker = Marker()
                    marker.header.frame_id = "base"
                    marker.header.stamp = self.get_clock().now().to_msg()
                    marker.ns = "slam_points"
                    marker.id = i
                    marker.type = Marker.SPHERE  # Use a cube to represent the marker
                    marker.action = Marker.ADD
                    marker.pose.position.x = translation[0]
                    marker.pose.position.y = translation[1]
                    marker.pose.position.z = translation[2]
                    marker.pose.orientation.x = 0.0
                    marker.pose.orientation.y = 0.0
                    marker.pose.orientation.z = 0.0
                    marker.pose.orientation.w = 1.0
                    marker.scale.x = 0.3
                    marker.scale.y = 0.3
                    marker.scale.z = 0.4
                    marker.color.a = 1.0
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker_array.markers.append(marker)

                self.marker_publisher.publish(marker_array)  # Publish the marker array
                self.get_logger().info('Publishing Markers')

                # Create a PointCloud2 message
                msg = PointCloud2()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = 'base'
                points = np.array(mapp.points)

                msg.height = 1
                msg.width = points.shape[0]

                # Define the fields for the point cloud
                msg.fields = [
                    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1)
                ]
                msg.is_bigendian = False
                msg.point_step = 12
                msg.row_step = msg.point_step * points.shape[0]
                msg.is_dense = True
                msg.data = np.asarray(points, np.float32).tobytes()

                self.pcd_publisher.publish(msg)  # Publish the point cloud
                self.get_logger().info('Publishing point cloud')

                # Convert the image to a ROS Image message and publish it
                image_message = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                self.feature_publisher.publish(image_message)
                self.get_logger().info('Published Feature Extraction Image')

                self.broadcast_transformations(mapp)

            else:
                self.get_logger().warn("Function returned None, cannot unpack")

def main(args=None):
    rclpy.init(args=args)  # Initialize the rclpy library

    minimal_subscriber = SlamPublisher()  # Create an instance of the SlamPublisher node

    rclpy.spin(minimal_subscriber)  # Spin the node to keep it running

    # Destroy the node explicitly (optional - otherwise it will be done automatically)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()  # Shutdown the rclpy library

if __name__ == '__main__':
    main()
