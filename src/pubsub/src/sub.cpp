#include <chrono>
#include <functional>
#include <memory>
#include <string>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

using namespace std::chrono_literals;
using namespace std::placeholders;

class Subscriber : public rclcpp::Node
{
    public:
        Subscriber() : Node("sub")
        {
            sub_ = this->create_subscription<std_msgs::msg::String>(
                "topic",
                10,
                std::bind(&Subscriber::topic_callback, this, _1)
            );
        }
    private:
       rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_;

        void topic_callback(const std_msgs::msg::String::SharedPtr msg)
        {
            RCLCPP_INFO(this->get_logger(), msg->data.c_str());
        }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<Subscriber>());
    rclcpp::shutdown();
    return 0;
}