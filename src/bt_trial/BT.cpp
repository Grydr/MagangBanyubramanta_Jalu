#include <iostream>
#include <chrono>
#include <behaviortree_cpp_v3/bt_factory.h>
#include <behaviortree_cpp_v3/action_node.h>

using namespace BT;
using namespace std::chrono;

class ApproachObject : public BT::SyncActionNode
{
public:
  ApproachObject(const std::string& name) :
      BT::SyncActionNode(name, {})
  {}

  BT::NodeStatus tick() override
  {
    std::cout << "ApproachObject: " << this->name() << std::endl;
    std::this_thread::sleep_for(seconds(5));
    return BT::NodeStatus::SUCCESS;
  }
};

BT::NodeStatus CheckBattery()
{
  std::cout << "[ Battery: OK! ]" << std::endl;
  return BT::NodeStatus::SUCCESS;
}

class GripperInterface
{
public:
  GripperInterface(): _open(true) {}
    
  BT::NodeStatus open() 
  {
    _open = true;
    std::cout << "--- Gripper is opening ---" << std::endl;
    return NodeStatus::SUCCESS;
  }

  BT::NodeStatus close() 
  {
    std::cout << "--- Gripper is closing ---" << std::endl;
    _open = false;
    return NodeStatus::SUCCESS;
  }

private:
  bool _open;
};

int main() {
    BehaviorTreeFactory factory;

    factory.registerNodeType<ApproachObject>("ApproachObject");
    factory.registerSimpleCondition("CheckBattery", std::bind(CheckBattery));

    GripperInterface gripper;
    factory.registerSimpleAction("OpenGripper", std::bind(&GripperInterface::open, &gripper) );
    factory.registerSimpleAction("CloseGripper", std::bind(&GripperInterface::close, &gripper) );

    auto tree = factory.createTreeFromFile("./../package.xml");
    tree.tickRoot();

    return 0;
}