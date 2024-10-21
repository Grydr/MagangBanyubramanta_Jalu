#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "interfaces/srv/add.hpp"

void add(const std::shared_ptr<interfaces::srv::Add::Request> req,
         std::shared_ptr<interfaces::srv::Add::Response> res)
{
    res->sum = req->a + req->b;
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Requested %d + %d, Responded %d", req->a, req->b, res->sum);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>("add_server");

    rclcpp::Service<interfaces::srv::Add>::SharedPtr server = node->create_service<interfaces::srv::Add>("add", &add);

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}