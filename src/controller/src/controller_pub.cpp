#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "sensor_msgs/msg/joy.hpp"

using namespace std::chrono_literals;

class ControllerPub : public rclcpp::Node {
   public:
      ControllerPub() : Node("controller_pub") {
         pub_ = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);
         sub_ = this->create_subscription<sensor_msgs::msg::Joy>("joy", 10, std::bind(&ControllerPub::controller_callback, this, std::placeholders::_1));
         RCLCPP_INFO(this->get_logger(), "Controller Pub started");
      }
   private:
      void controller_callback(const sensor_msgs::msg::Joy::SharedPtr msg) {
         geometry_msgs::msg::Twist twist_;
         twist_.linear.x = msg->axes[0]; // left stick left & right
         twist_.linear.y = msg->axes[1]; // left stick up & down
         twist_.linear.z = msg->axes[4]; // right stick up & down
         twist_.angular.z = msg->axes[3]; // right stick left & right
 
         pub_->publish(twist_);
      }
      rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
      rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_;

};

int main(int argc, char *argv[]) {
   rclcpp::init(argc, argv);
   rclcpp::spin(std::make_shared<ControllerPub>()); 
   rclcpp::shutdown();
   return 0;
}