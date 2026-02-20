#ifndef KINEMATIC_CONTROLLER_HPP
#define KINEMATIC_CONTROLLER_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>
#include <pinocchio/algorithm/compute-all-terms.hpp>   // IMPORTANT for computeAllTerms

#include <Eigen/Dense>
#include <string>

class KinematicController : public rclcpp::Node
{
public:
  KinematicController();

private:
  // Init
  void init();
  bool readParameters();

  // Callbacks
  void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void referenceTwistCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
  void referenceJointVelCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg); // NEW

  // Computation
  void computeForwardKinematics();
  void computeJacobian();
  void computeInverseKinematics();
  void publishFeedback();

  // Pinocchio
  pinocchio::Model model_;
  pinocchio::Data data_;
  int hand_id_{-1};
  int dim_joints_{0};

  // Robot state
  Eigen::VectorXd joint_pos_;
  Eigen::VectorXd joint_vel_;
  Eigen::Vector3d fbk_task_pos_;
  Eigen::Vector3d fbk_task_vel_;

  // IK variables
  Eigen::MatrixXd jacobian_;     // 6x7
  Eigen::MatrixXd J_linear_;     // 3x7
  Eigen::MatrixXd J_pinv_;       // 7x3
  Eigen::Vector3d ref_task_vel_; // 3x1
  Eigen::VectorXd cmd_joint_vel_;// 7x1

  // Redundancy (Step V)
  Eigen::MatrixXd N_;            // 7x7 nullspace projector
  Eigen::VectorXd ref_joint_vel_;// 7x1 reference from joint planner
  bool with_redundancy_{false};  // NEW

  // Flags
  bool first_joint_received_{false};
  bool first_twist_received_{false};
  bool first_joint_ref_received_{false}; // NEW

  // ROS interfaces
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr joint_ref_sub_; // NEW

  rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  // Parameters
  std::string urdf_file_name_;
  double publish_rate_{500.0};
};

#endif
