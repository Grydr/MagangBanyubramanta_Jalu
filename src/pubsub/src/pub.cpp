#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;

class Publisher : public rclcpp::Node
{
    public:
        Publisher() : Node("pub")
        {
            timer_ = this->create_wall_timer(
                500ms,
                std::bind(&Publisher::timer_callback, this)
            );

            pub_ = this->create_publisher<std_msgs::msg::String>("topic", 10);
        }
    private:
        void timer_callback()
        {
            auto msg = std_msgs::msg::String();
            msg.data = "Halo dunia!";

            pub_->publish(msg);
        }
        
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Publisher>());
    rclcpp::shutdown();
    return 0;
}