import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import lanelet2
from lanelet2.projection import UtmProjector

class Lanelet2Publisher(Node):
    def __init__(self):
        super().__init__('lanelet2_publisher')
        self.publisher_ = self.create_publisher(MarkerArray, 'lanelet2_map', 10)
        self.timer = self.create_timer(1.0, self.timer_callback)
        
        # Load the Lanelet2 map with LatLonProjector
        projector = UtmProjector(lanelet2.io.Origin(0,0))
        self.lanelet_map = lanelet2.io.load("src/vehicle_ctrl/maps/Town01_lanelet.osm", projector)
        print("Map loaded ...")
        
    def timer_callback(self):
        marker_array = MarkerArray()
        marker_id = 0  # Initialize marker ID
        # print("=----------------------------")
        # print("self.lanelet_map: ",dir(self.lanelet_map))
        # print("laneletLayer: ",type(self.lanelet_map.laneletLayer))
        for lanelet_ in self.lanelet_map.laneletLayer:
            marker = Marker()
            marker.header.frame_id = "map"#"ego_vehicle/lidar"
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.3
            marker.color.a = 0.5
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.ns = "lanelet"
            marker.id = marker_id  # Assign unique ID
            # print("lanelet_: ", dir(lanelet_.attributes))
            if "subtype" not in lanelet_.attributes.keys():
                # print("\n\n\n\n\n\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n\n\n\n\n\n")
                for point in lanelet2.geometry.to2D(lanelet_.leftBound):
                    marker.points.append(Point(x=point.x, y=point.y, z=0.0))

                for point in lanelet2.geometry.to2D(lanelet_.rightBound):
                    marker.points.append(Point(x=point.x, y=point.y, z=0.0))

            marker_array.markers.append(marker)
            marker_id += 1

            
        self.publisher_.publish(marker_array)

def main(args=None):
    rclpy.init(args=args)
    node = Lanelet2Publisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
