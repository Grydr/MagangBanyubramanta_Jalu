#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include "interfaces/msg/commands.hpp"

using namespace std::chrono_literals;

class ControllerNode : public rclcpp::Node {

   private:
      rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_;
      rclcpp::Publisher<interfaces::msg::Commands>::SharedPtr pub_;

      void controller_callback(const sensor_msgs::msg::Joy msg) {
         auto cmd = interfaces::msg::Commands();
         cmd.x_cmd = msg.axes[0];

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