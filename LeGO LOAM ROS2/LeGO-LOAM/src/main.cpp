#include "rclcpp/rclcpp.hpp"
#include "featureAssociation.h"
#include "imageProjection.h"
#include "mapOptimization.h"
#include "transformFusion.h"

int main(int argc, char** argv) {
  Channel<ProjectionOut> projection_out_channel(true);
  Channel<AssociationOut> association_out_channel(false);

  rclcpp::init(argc, argv);

  // Create nodes
  auto IP = std::make_shared<ImageProjection>("image_projection", projection_out_channel);
  auto FA = std::make_shared<FeatureAssociation>("feature_association", projection_out_channel, association_out_channel);
  auto MO = std::make_shared<MapOptimization>("map_optimization", association_out_channel);
  auto TF = std::make_shared<TransformFusion>("transform_fusion");

  RCLCPP_INFO(IP->get_logger(), "\033[1;32m---->\033[0m Started.");
  RCLCPP_INFO(FA->get_logger(), "\033[1;32m---->\033[0m Started.");
  RCLCPP_INFO(MO->get_logger(), "\033[1;32m---->\033[0m Started.");
  RCLCPP_INFO(TF->get_logger(), "\033[1;32m---->\033[0m Started.");

  // Use 4 threads
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);//been modify to galactic syntax
  executor.add_node(IP);
  executor.add_node(FA);
  executor.add_node(MO);
  executor.add_node(TF);
  executor.spin();

  // Must be called to cleanup threads
  rclcpp::shutdown();

  return 0;
}
