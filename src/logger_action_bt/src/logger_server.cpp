#include <functional>
#include <memory>
#include <thread>

#include "interfaces/action/logger.hpp"
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"

class KeyActionServer : public rclcpp::Node {
 public:
   using Logger = interfaces::action::Logger;
   using GoalHandleLogger = rclcpp_action::ServerGoalHandle<Logger>;

   KeyActionServer(const rclcpp::NodeOptions &options) : Node("key_action_server", options) {
      using namespace std::placeholders;

      this->action_server_ = rclcpp_action::create_server<Logger>(this, "key_teleop", std::bind())
   }
};