#include <functional>
#include <memory>
#include <string>

#include <gazebo/common/common.hh>
#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <gazebo_ros/conversions/geometry_msgs.hpp>
#include <gazebo_ros/node.hpp>
#include <ignition/math/Vector3.hh>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "pid.hpp"

namespace gazebo {
class ROV_Controller : public ModelPlugin {
 public:
   void Load(physics::ModelPtr _model, sdf::ElementPtr _sdf) {
      this->world = _model->GetWorld();
      this->model = _model;
      this->updateConnection = event::Events::ConnectWorldUpdateBegin(std::bind(&ROV_Controller::OnUpdate, this));
      this->prevTime = world->SimTime();

      this->node = gazebo_ros::Node::Get(_sdf);
      RCLCPP_INFO(this->node->get_logger(), "Model plugin initialized");

      this->cmd_vel_sub = this->node->create_subscription<geometry_msgs::msg::Twist>(
          "cmd_vel", 10, std::bind(&ROV_Controller::OnCmdVel, this, std::placeholders::_1));
      this->odom_pub = this->node->create_publisher<nav_msgs::msg::Odometry>("odom", 10);

      this->tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(this->node);
   }

   void OnUpdate() {
      common::Time currTime = this->world->SimTime();
      double dt = (currTime - prevTime).Double();
      prevTime = currTime;

      ignition::math::Pose3d pose = this->model->WorldPose();
      // double x = pose.Pos().X(), y = pose.Pos().Y(), z = pose.Pos().Z(), roll = pose.Rot().Roll(),
      //        pitch = pose.Rot().Pitch(), yaw = pose.Rot().Yaw();

      nav_msgs::msg::Odometry odom;
      odom.header.stamp.sec = currTime.sec;
      odom.header.stamp.nanosec = currTime.nsec;
      odom.header.frame_id = "odom";
      odom.child_frame_id = "base_link";

      ignition::math::Vector3d current_linear_vel = this->model->WorldLinearVel();
      ignition::math::Vector3d current_angular_vel = this->model->WorldAngularVel();

      double control_linear_x = pid_linear_x.compute(target_linear_vel.X(), current_linear_vel.X(), dt);
      double control_linear_y = pid_linear_y.compute(target_linear_vel.Y(), current_linear_vel.Y(), dt);
      double control_linear_z = pid_linear_z.compute(target_linear_vel.Z(), current_linear_vel.Z(), dt);

      double control_angular_roll = pid_angular_roll.compute(target_angular_vel.X(), current_angular_vel.X(), dt);
      double control_angular_pitch = pid_angular_pitch.compute(target_angular_vel.Y(), current_angular_vel.Y(), dt);
      double control_angular_yaw = pid_angular_yaw.compute(target_angular_vel.Z(), current_angular_vel.Z(), dt);

      this->model->SetLinearVel(ignition::math::Vector3d(control_linear_x, control_linear_y, control_linear_z));
      this->model->SetAngularVel(
          ignition::math::Vector3d(control_angular_roll, control_angular_pitch, control_angular_yaw));

      odom.pose.pose = gazebo_ros::Convert<geometry_msgs::msg::Pose>(pose);
      // odom.twist.twist.linear.x = (x_vel * cosf(yaw) + y_vel * sinf(yaw));
      // odom.twist.twist.linear.y = (y_vel * cosf(yaw) - y_vel * sinf(yaw));
      // odom.twist.twist.linear.z = z_vel;
      odom.twist.twist.linear.x = current_linear_vel.X();
      odom.twist.twist.linear.y = current_linear_vel.Y();
      odom.twist.twist.linear.z = current_linear_vel.Z();
      odom.twist.twist.angular.x = current_angular_vel.X();
      odom.twist.twist.angular.y = current_angular_vel.Y();
      odom.twist.twist.angular.z = current_angular_vel.Z();

      double cov[36] = {1e-3, 0, 0, 0,   0, 0, 0, 1e-3, 0, 0, 0,   0, 0, 0, 1e6, 0, 0, 0,
                        0,    0, 0, 1e6, 0, 0, 0, 0,    0, 0, 1e6, 0, 0, 0, 0,   0, 0, 1e3};

      memcpy(&odom.pose.covariance, cov, sizeof(double) * 36);
      memcpy(&odom.twist.covariance, cov, sizeof(double) * 36);

      this->odom_pub->publish(odom);

      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp.sec = currTime.sec;
      tf.header.stamp.nanosec = currTime.nsec;
      tf.header.frame_id = "odom";
      tf.child_frame_id = "base_link";
      tf.transform = gazebo_ros::Convert<geometry_msgs::msg::Transform>(odom.pose.pose);

      tf_broadcaster->sendTransform(tf);
   }

   void OnCmdVel(const geometry_msgs::msg::Twist &msg) {
      // ignition::math::Pose3d pose = this->model->WorldPose();

      // this->model->SetLinearVel(ignition::math::Vector3d(
      //     msg.linear.x * cosf(pose.Rot().Yaw()) - msg.linear.y * sinf(pose.Rot().Yaw()),
      //     msg.linear.y * cosf(pose.Rot().Yaw()) + msg.linear.x * sinf(pose.Rot().Yaw()), msg.linear.z));
      // this->model->SetAngularVel(ignition::math::Vector3d(msg.angular.x, msg.angular.y, msg.angular.z));

      // PID set
      this->target_angular_vel.Set(msg.angular.x, msg.angular.y, msg.angular.z);
      this->target_linear_vel.Set(msg.linear.x, msg.linear.y, msg.linear.z);
   }

 private:
   physics::WorldPtr world;
   physics::ModelPtr model;
   common::Time prevTime;
   event::ConnectionPtr updateConnection;
   std::shared_ptr<rclcpp::Node> node;
   rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_sub;
   rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub;
   std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

   // variabel tambahan
   PID pid_linear_x{1.0, 0.0, 0.0};
   PID pid_linear_y{1.0, 0.0, 0.0};
   PID pid_linear_z{1.0, 0.0, 0.0};
   PID pid_angular_roll{1.0, 0.0, 0.0};
   PID pid_angular_pitch{1.0, 0.0, 0.0};
   PID pid_angular_yaw{1.0, 0.0, 0.0};

   ignition::math::Vector3d target_linear_vel;
   ignition::math::Vector3d target_angular_vel;
};

GZ_REGISTER_MODEL_PLUGIN(ROV_Controller)
} // namespace gazebo