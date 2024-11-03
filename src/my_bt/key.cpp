#include <iostream>
#include <termios.h>
#include <fcntl.h>
#include <unistd.h>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/action_node.h>

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

int main(int argc, char **argv) {
   BT::BehaviorTreeFactory factory;

   factory.registerNodeType<InLogger>("InLogger");
   factory.registerNodeType<OutLogger>("OutLogger");

   auto tree = factory.createTreeFromFile("./../package.xml");
   while(true) {
      tree.tickRoot();
      usleep(100000);
   }
   return 0;
}