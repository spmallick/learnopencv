import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
# from tf_transformations import quaternion_from_euler
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import math

class CarData(Node):

    def __init__(self):
        super().__init__('car_data')
        
        # Simulation state from parameter
        self.declare_parameter('simulation_state', True)
        self.simulation_state = self.get_parameter('simulation_state').get_parameter_value().bool_value

        # Publishers and Subscribers
        if self.simulation_state:
            self.car_data_publisher = self.create_publisher(PoseStamped, '/pose', 1)
            self.car_odom_publisher = self.create_publisher(Odometry, '/odometry/filtered/global', 1)
            self.car_data_subscriber = self.create_subscription(
                Odometry,
                '/carla/ego_vehicle/odometry',
                self.sim_callback,
                1
            )
            self.br = TransformBroadcaster(self)
        else:
            self.get_logger().info('Simulation state is false')

    def sim_callback(self, msg):
        self.get_logger().info(f'Sim Callback: {msg.header.stamp}')

        # Prepare PoseStamped message
        car_data = PoseStamped()
        car_data.header.stamp = self.get_clock().now().to_msg()
        car_data.header.frame_id = 'map'
        car_data.pose = msg.pose.pose
        self.car_data_publisher.publish(car_data)

        # Prepare transform for broadcasting
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = 'map'
        transform.child_frame_id = 'base_link'
        transform.transform.translation.x = car_data.pose.position.x
        transform.transform.translation.y = car_data.pose.position.y
        transform.transform.translation.z = car_data.pose.position.z

        quat = car_data.pose.orientation
        transform.transform.rotation = quat

        self.br.sendTransform(transform)

        # Prepare and publish Odometry message
        car_odom = msg
        car_odom.child_frame_id = 'base_link'
        self.car_odom_publisher.publish(car_odom)

def main(args=None):
    rclpy.init(args=args)
    car_data_node = CarData()
    
    try:
        rclpy.spin(car_data_node)
    except KeyboardInterrupt:
        pass
    
    car_data_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
