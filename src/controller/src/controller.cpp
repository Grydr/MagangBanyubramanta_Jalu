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
         2        LT
         5        RT
*/

class ControllerNode : public rclcpp::Node {

   private:
      rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr sub_;
      rclcpp::Publisher<interfaces::msg::Commands>::SharedPtr pub_;
      int x_range = -250;
      int y_range = 250;
      int yaw_range = -180;
      int depth_range = 10;
      int temp_depth = 0;
      int temp_yaw = 0;
      
      void controller_callback(const sensor_msgs::msg::Joy &msg) {
         // use axes[4] for depth
         // if axes[4] is 1(up) then decrement depth;
            if (msg.axes[4] > 0) {
               if(temp_depth == 0) {
                  temp_depth = temp_depth + 1 ;
               }
               temp_depth--;
            }
         // if axes[4] is -1(down) then increment depth;
            if (msg.axes[4] < 0) {
               if(temp_depth == 10) {
                  temp_depth = temp_depth - 1;
               }
               temp_depth++;
            }
         // yaw right
            if ((msg.axes[3] * -1) > 0) {
               temp_yaw += msg.axes[3] * yaw_range;
               if (temp_yaw > 180) {
                  temp_yaw = yaw_range;
               }
            }
         // yaw left
            if (msg.axes[3] > 0) {
               temp_yaw += msg.axes[3] * yaw_range;
               // temp_yaw--;
               if (temp_yaw < -180) {
                  temp_yaw = yaw_range * -1;
               }
            }

            auto cmd = interfaces::msg::Commands();
            cmd.x_cmd = msg.axes[0] * x_range; // move left & right using left stick
            cmd.y_cmd = msg.axes[1] * y_range; // move up & down using left stick
            cmd.depth = temp_depth;
            cmd.yaw = temp_yaw;

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