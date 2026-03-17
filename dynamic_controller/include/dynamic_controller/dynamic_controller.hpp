#ifndef DYNAMIC_CONTROLLER_HPP
#define DYNAMIC_CONTROLLER_HPP

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <highlevel_interfaces/srv/set_kd.hpp>
#include <pinocchio/algorithm/crba.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/rnea.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <Eigen/Dense>

#include <memory>
#include <string>

class DynamicController : public rclcpp::Node
{
public:
  DynamicController();

private:
  bool readParameters();
  void init();

  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void referencePoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg);
  void referenceTwistCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void referenceJointPosCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void referenceJointVelCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg);
  void setLinearKDCallback(
    const std::shared_ptr<highlevel_interfaces::srv::SetKD::Request> request,
    std::shared_ptr<highlevel_interfaces::srv::SetKD::Response> response);
  void setJointKDCallback(
    const std::shared_ptr<highlevel_interfaces::srv::SetKD::Request> request,
    std::shared_ptr<highlevel_interfaces::srv::SetKD::Response> response);

  void computeKinematicsAndDynamics();
  void publishFeedback();
  void computeAndPublishCommand();
  Eigen::MatrixXd pseudoInverse(const Eigen::MatrixXd & matrix) const;

  pinocchio::Model model_;
  pinocchio::Data data_;
  pinocchio::FrameIndex hand_frame_id_{0};
  int dim_joints_{0};

  Eigen::VectorXd joint_pos_;
  Eigen::VectorXd joint_vel_;
  Eigen::Vector3d fbk_task_pos_;
  Eigen::Vector3d fbk_task_vel_;
  Eigen::Vector3d ref_task_pos_;
  Eigen::Vector3d ref_task_vel_;
  Eigen::VectorXd ref_joint_pos_;
  Eigen::VectorXd ref_joint_vel_;
  Eigen::VectorXd cmd_joint_torque_;

  Eigen::MatrixXd jacobian_;
  Eigen::MatrixXd jacobian_dot_;
  Eigen::MatrixXd j_linear_;
  Eigen::MatrixXd jdot_linear_;
  Eigen::MatrixXd mass_matrix_;
  Eigen::MatrixXd lambda_;
  Eigen::MatrixXd nullspace_projection_;
  Eigen::VectorXd nonlinear_effects_;

  bool with_redundancy_{false};
  bool first_joint_received_{false};
  bool first_pose_ref_received_{false};
  bool first_twist_ref_received_{false};
  bool first_joint_pos_ref_received_{false};
  bool first_joint_vel_ref_received_{false};

  std::string urdf_file_name_;
  double publish_rate_{500.0};
  double linear_k_{5.0};
  double linear_d_{1.0};
  double joint_k_{100.0};
  double joint_d_{20.0};

  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_ref_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_ref_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pos_ref_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_ref_sub_;

  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr torque_pub_;

  rclcpp::Service<highlevel_interfaces::srv::SetKD>::SharedPtr linear_kd_srv_;
  rclcpp::Service<highlevel_interfaces::srv::SetKD>::SharedPtr joint_kd_srv_;
  rclcpp::TimerBase::SharedPtr timer_;
};

#endif
