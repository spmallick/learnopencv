# https://carla.readthedocs.io/projects/ros-bridge/en/latest/carla_waypoint/

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

class GoalPoseRepublisher(Node):
    def __init__(self, role_name):
        super().__init__('goal_remap')
        self.role_name = role_name
        # Subscribe to the /goal_pose topic
        self.subscription = self.create_subscription(
            PoseStamped,
            '/goal_pose',
            self.goal_pose_callback,
            10
        )

        # Publisher to /carla/<ROLE_NAME>/goal
        self.publisher = self.create_publisher(
            PoseStamped,
            f'/carla/{self.role_name}/goal',
            10
        )
        print("Re-mapping started ...")
        
    def goal_pose_callback(self, msg):
        # Log the received goal pose
        self.get_logger().info(f'Received goal pose: {msg}')

        # Create a new PoseStamped message to publish
        new_msg = PoseStamped()

        # Copy the header from the received message
        new_msg.header.stamp = self.get_clock().now().to_msg()  # Use the current time for the header
        new_msg.header.frame_id = 'map'  # You can change this to the appropriate frame id

        # Modify the pose values
        new_msg.pose.position.x = msg.pose.position.x
        new_msg.pose.position.y = msg.pose.position.y
        new_msg.pose.position.z = 0.2  #msg.pose.position.z  # Retain the original Z value

        # Copy the orientation directly from the received message
        new_msg.pose.orientation = msg.pose.orientation

        # Republish the goal pose to the Carla-specific topic
        self.publisher.publish(new_msg)
        self.get_logger().info(f'Published goal pose to /carla/{self.role_name}/goal')



def main(args=None):
    rclpy.init(args=args)

    # Replace 'my_role_name' with your actual role name in Carla
    role_name = 'ego_vehicle'

    goal_pose_republisher = GoalPoseRepublisher(role_name)

    try:
        rclpy.spin(goal_pose_republisher)
    except KeyboardInterrupt:
        pass
    finally:
        goal_pose_republisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
