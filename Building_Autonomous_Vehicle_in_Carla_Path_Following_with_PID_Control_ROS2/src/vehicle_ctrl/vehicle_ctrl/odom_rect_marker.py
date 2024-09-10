import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"
RESET = "\033[0m"
role_name = "ego_vehicle"


class CarlaVehicleNode(Node):
    def __init__(self):
        super().__init__('carla_vehicle_node')
        
        self.odom_received = False
        self.waypoints_received = False
        
        # Subscription to odometry topic
        self.subscription_odom = self.create_subscription(
            Odometry,
            f'/carla/{role_name}/odometry',
            self.odometry_callback,
            10)
        
        # Subscription to waypoints topic (Path)
        self.subscription_waypoints = self.create_subscription(
            Path,
            f'/carla/{role_name}/waypoints',
            self.waypoints_callback,
            10)
        self.veh_marker_pub = self.create_publisher(MarkerArray,
                                                    f'/carla/{role_name}/veh_frame',
                                                    10)

        self.marker_array = MarkerArray()
        self.veh_width = 3
        self.veh_height = 3
        self.itr = 0

    def marker_msg(self, position):
        rectangle_marker = Marker()
        rectangle_marker.header.frame_id = "map"
        rectangle_marker.header.stamp = self.get_clock().now().to_msg()
        rectangle_marker.ns = "rectangle"+str(self.itr)
        rectangle_marker.id = self.itr
        rectangle_marker.type = Marker.LINE_STRIP
        rectangle_marker.action = Marker.ADD
        rectangle_marker.scale.x = 0.5  # Line width
        
        rectangle_marker.color.r = 0.0
        rectangle_marker.color.g = 1.0
        rectangle_marker.color.b = 0.0
        rectangle_marker.color.a = 1.0

        # Define the four corners of the rectangle

        tr = Point(x=(position.x + self.veh_width/2), y=(position.y + self.veh_height/2), z=0.1)
        tl = Point(x=(position.x - self.veh_width/2), y=(position.y + self.veh_height/2), z=0.1)
        br = Point(x=(position.x + self.veh_width/2), y=(position.y - self.veh_height/2), z=0.1)
        bl = Point(x=(position.x - self.veh_width/2), y=(position.y - self.veh_height/2), z=0.1)

        rectangle_marker.points.append(tr)
        rectangle_marker.points.append(tl)
        rectangle_marker.points.append(bl)
        rectangle_marker.points.append(br)
        rectangle_marker.points.append(tr)  # Close the rectangle
        return rectangle_marker

    def odometry_callback(self, msg):
        # if self.odom_received:
        #     return  # Exit if already processed

        # Process odometry data
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        velocity = msg.twist.twist.linear
        
        self.get_logger().info(f'Position: x={position.x}, y={position.y}, z={position.z}')
        self.get_logger().info(f'Orientation: x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}')
        self.get_logger().info(f'Velocity: x={velocity.x}, y={velocity.y}, z={velocity.z}')
        
        self.odom_received = True  # Mark as received

        
        rectangle_marker = self.marker_msg(position)
        # Add the rectangle marker to the marker array
        self.marker_array.markers.append(rectangle_marker)

        # Publish the marker array
        self.veh_marker_pub.publish(self.marker_array)
        self.get_logger().info(f"Publishing rectangle marker array | itr: {self.itr}")  
        self.itr+=1


    def waypoints_callback(self, msg):
        
        if self.waypoints_received:
            return  # Exit if already processed

        # Process waypoints data (Path)
        print("\n\n=----------------------------------------------------------------")
        for pose_stamped in msg.poses:
            position = pose_stamped.pose.position
            self.get_logger().info(f'Waypoint Position: x={position.x}, y={position.y}, z={position.z}')

            
        start = msg.poses[0].pose.position
        end = msg.poses[1].pose.position

        # position = start.pose.position
        print("\n\n=----------------------------------------------------------------")
        self.get_logger().info(f'{GREEN}Waypoint Position: x={start.x}, y={start.y}, z={start.z}{RESET}')
        self.get_logger().info(f'{GREEN}Waypoint Position: x={end.x}, y={end.y}, z={end.z}{RESET}')


        
        self.waypoints_received = True  # Mark as received

def main(args=None):
    rclpy.init(args=args)

    carla_vehicle_node = CarlaVehicleNode()

    rclpy.spin(carla_vehicle_node)

    # Cleanup
    carla_vehicle_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
