#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "interfaces/msg/commands.hpp"

using namespace std::chrono_literals;

/*
      /cmd_vel
      x_cmd = [-250] to [250]
      y_cmd = [-250] to [250]
      yaw = [-180] to [180]
      depth = [0] to [10]
*/

/*
      axes index  action
         0        Left_Right_StickLeft
         1        Up_Down_StickLeft
         3        Left_Right_StickRight
         4        Up_Down_StickRight
*/

class ControllerNode : public rclcpp::Node {

   private:
      rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_;
      rclcpp::Publisher<interfaces::msg::Commands>::SharedPtr pub_;
      float temp_depth;
      float xy_range = 250;
      float yaw_range = 180;
      float depth_range = 10;

      void controller_callback(const sensor_msgs::msg::Joy msg) {
         auto cmd = interfaces::msg::Commands();
         cmd.x_cmd = msg.axes[0] * xy_range; // move left & right using left stick
         cmd.y_cmd = msg.axes[1] * xy_range; // move up & down using left stick
         temp_depth = std::max(0.0f, msg.axes[4]);
         cmd.depth = temp_depth * depth_range; // move up & down using right stick
         cmd.yaw = msg.axes[3] * yaw_range; // move left & right using right stick

         // RCLCPP_INFO(this->get_logger(), "Parsing controller data");
         pub_->publish(cmd);
      }

   public:
      ControllerNode() : Node("controller_node") {
         pub_ = this->create_publisher<interfaces::msg::Commands>("cmd_vel", 10);
         sub_ = this->create_subscription<sensor_msgs::msg::Joy>("joy", 10, std::bind(&ControllerNode::controller_callback, this, std::placeholders::_1));
      }
};

int main(int argc, char *argv[]) {
   rclcpp::init(argc, argv);
   rclcpp::spin(std::make_shared<ControllerNode>()); 
   rclcpp::shutdown();
   return 0;
}