#include <iostream>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/action_node.h>
#include "rclcpp/rclcpp.hpp"

using namespace std::chrono;

class InLogger : public BT::SyncActionNode {
   public:
      InLogger(const std::string &name) : BT::SyncActionNode(name, {}) {}

      BT::NodeStatus tick() override {
         if (isKeyPressed('f')) {
            return BT::NodeStatus::SUCCESS;
         } else {
            return BT::NodeStatus::FAILURE;
         }
      }

   private:
      bool isKeyPressed(int key) {
         struct termios oldt, newt;
         int oldf;
         char ch;
         bool keyPressed = false;

         tcgetattr(STDIN_FILENO, &oldt);
         newt = oldt;
         newt.c_lflag &= ~(ICANON | ECHO);
         tcsetattr(STDIN_FILENO, TCSANOW, &newt);
         oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
         fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

         ch = getchar();

         if (ch == key) {
            keyPressed = true;
         }

         tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
         fcntl(STDIN_FILENO, F_SETFL, oldf);

         return keyPressed;
   }
};

class OutLogger : public BT::SyncActionNode {
   public:
      OutLogger(const std::string &name) : BT::SyncActionNode(name, {}) {}

      BT::NodeStatus tick() override {
         std::cout << "Key Pressed!" << std::endl;
         return BT::NodeStatus::SUCCESS;
      }
};

class KeyLogger : public rclcpp::Node {
   public:
      KeyLogger() : Node("key_logger_node") {
         factory.registerNodeType<InLogger>("InLogger");
         factory.registerNodeType<OutLogger>("OutLogger");

         tree = factory.createTreeFromFile("src/key_logger/bt_package.xml");

         timer_ = this->create_wall_timer(100ms, std::bind(&KeyLogger::tick_tree, this));
      }

   private:
      BT::BehaviorTreeFactory factory;
      BT::Tree tree;
      rclcpp::TimerBase::SharedPtr timer_;

      void tick_tree() {
         tree.tickRoot();
      }
};

int main(int argc, char **argv) {
   rclcpp::init(argc, argv);
   rclcpp::spin(std::make_shared<KeyLogger>());
   rclcpp::shutdown();
   return 0;
}