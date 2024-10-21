#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "interfaces/srv/add.hpp"

using namespace std::chrono_literals;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);

    std::shared_ptr<rclcpp::Node> node = std::make_shared<rclcpp::Node>("add_client");
    rclcpp::Client<interfaces::srv::Add>::SharedPtr client = node->create_client<interfaces::srv::Add>("add");

    auto req = std::make_shared<interfaces::srv::Add::Request>();
    req->a = atoi(argv[1]);
    req->b = atoi(argv[2]);

    while (!client->wait_for_service(1s))
    {
        if (!rclcpp::ok())
        {
            RCLCPP_ERROR(node->get_logger(), "Terminated");
            return -1;
        }

        RCLCPP_INFO(node->get_logger(), "Waiting for service");
    }

    auto res = client->async_send_request(req);

    if (rclcpp::spin_until_future_complete(node, res) == rclcpp::FutureReturnCode::SUCCESS)
        RCLCPP_INFO(node->get_logger(), "Sum: %d", res.get()->sum);
    else
        RCLCPP_ERROR(node->get_logger(), "Failed to call service");

    rclcpp::shutdown();

    return 0;
}