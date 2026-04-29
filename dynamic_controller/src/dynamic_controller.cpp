#include "dynamic_controller/dynamic_controller.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>

DynamicController::DynamicController()
: Node("dynamic_controller")
{
  if (!readParameters()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to read parameters, shutting down.");
    rclcpp::shutdown();
    return;
  }

  init();

  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", qos,
    std::bind(&DynamicController::jointStateCallback, this, std::placeholders::_1));
  pose_ref_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
    "/gen3/reference/pose", qos,
    std::bind(&DynamicController::referencePoseCallback, this, std::placeholders::_1));
  twist_ref_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "/gen3/reference/twist", qos,
    std::bind(&DynamicController::referenceTwistCallback, this, std::placeholders::_1));
  joint_pos_ref_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/gen3/reference/position", qos,
    std::bind(&DynamicController::referenceJointPosCallback, this, std::placeholders::_1));
  joint_vel_ref_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/gen3/reference/velocity", qos,
    std::bind(&DynamicController::referenceJointVelCallback, this, std::placeholders::_1));

  pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>("/gen3/feedback/pose", qos);
  twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/gen3/feedback/twist", qos);
  torque_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/joint_effort_controller/commands", qos);

  linear_kd_srv_ = this->create_service<highlevel_interfaces::srv::SetKD>(
    "/gen3/set_linear_kd",
    std::bind(&DynamicController::setLinearKDCallback, this, std::placeholders::_1, std::placeholders::_2));
  joint_kd_srv_ = this->create_service<highlevel_interfaces::srv::SetKD>(
    "/gen3/set_joint_kd",
    std::bind(&DynamicController::setJointKDCallback, this, std::placeholders::_1, std::placeholders::_2));

  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / publish_rate_),
    [this]() {
      if (!first_joint_received_) {
        return;
      }
      computeKinematicsAndDynamics();
      publishFeedback();
      computeAndPublishCommand();
    });
}

bool DynamicController::readParameters()
{
  this->declare_parameter<std::string>("urdf_file_name", "");
  this->declare_parameter<double>("publish_rate", 500.0);
  this->declare_parameter<bool>("with_redundancy", false);
  this->declare_parameter<double>("linear_k", 480.0);
  this->declare_parameter<double>("linear_d", 760.0);
  this->declare_parameter<double>("joint_k", 100.0);
  this->declare_parameter<double>("joint_d", 20.0);

  if (!this->get_parameter("urdf_file_name", urdf_file_name_) || urdf_file_name_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Parameter urdf_file_name is missing or empty");
    return false;
  }

  this->get_parameter("publish_rate", publish_rate_);
  this->get_parameter("with_redundancy", with_redundancy_);
  this->get_parameter("linear_k", linear_k_);
  this->get_parameter("linear_d", linear_d_);
  this->get_parameter("joint_k", joint_k_);
  this->get_parameter("joint_d", joint_d_);
  return true;
}

void DynamicController::init()
{
  pinocchio::urdf::buildModel(urdf_file_name_, model_, false);
  data_ = pinocchio::Data(model_);
  hand_frame_id_ = model_.getFrameId("bracelet_link");
  dim_joints_ = model_.nv;

  if (hand_frame_id_ == static_cast<pinocchio::FrameIndex>(model_.nframes)) {
    throw std::runtime_error("Could not find frame bracelet_link in the URDF model");
  }

  joint_pos_ = Eigen::VectorXd::Zero(dim_joints_);
  joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);
  fbk_task_pos_.setZero();
  fbk_task_vel_.setZero();
  ref_task_pos_.setZero();
  ref_task_vel_.setZero();
  ref_joint_pos_ = Eigen::VectorXd::Zero(dim_joints_);
  ref_joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);
  cmd_joint_torque_ = Eigen::VectorXd::Zero(dim_joints_);

  jacobian_ = Eigen::MatrixXd::Zero(6, dim_joints_);
  jacobian_dot_ = Eigen::MatrixXd::Zero(6, dim_joints_);
  j_linear_ = Eigen::MatrixXd::Zero(3, dim_joints_);
  jdot_linear_ = Eigen::MatrixXd::Zero(3, dim_joints_);
  mass_matrix_ = Eigen::MatrixXd::Zero(dim_joints_, dim_joints_);
  lambda_ = Eigen::MatrixXd::Zero(3, 3);
  nullspace_projection_ = Eigen::MatrixXd::Identity(dim_joints_, dim_joints_);
  nonlinear_effects_ = Eigen::VectorXd::Zero(dim_joints_);
}

void DynamicController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  const size_t npos = std::min(msg->position.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < npos; ++i) {
    joint_pos_[static_cast<int>(i)] = msg->position[i];
  }

  const size_t nvel = std::min(msg->velocity.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < nvel; ++i) {
    joint_vel_[static_cast<int>(i)] = msg->velocity[i];
  }
  for (size_t i = nvel; i < npos; ++i) {
    joint_vel_[static_cast<int>(i)] = 0.0;
  }

  if (!first_joint_received_) {
    first_joint_received_ = true;
  }
}

void DynamicController::referencePoseCallback(const geometry_msgs::msg::Pose::SharedPtr msg)
{
  ref_task_pos_ << msg->position.x, msg->position.y, msg->position.z;
  first_pose_ref_received_ = true;
}

void DynamicController::referenceTwistCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  ref_task_vel_ << msg->linear.x, msg->linear.y, msg->linear.z;
  first_twist_ref_received_ = true;
}

void DynamicController::referenceJointPosCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  const size_t n = std::min(msg->data.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < n; ++i) {
    ref_joint_pos_[static_cast<int>(i)] = msg->data[i];
  }
  for (size_t i = n; i < static_cast<size_t>(dim_joints_); ++i) {
    ref_joint_pos_[static_cast<int>(i)] = 0.0;
  }
  first_joint_pos_ref_received_ = true;
}

void DynamicController::referenceJointVelCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  const size_t n = std::min(msg->data.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < n; ++i) {
    ref_joint_vel_[static_cast<int>(i)] = msg->data[i];
  }
  for (size_t i = n; i < static_cast<size_t>(dim_joints_); ++i) {
    ref_joint_vel_[static_cast<int>(i)] = 0.0;
  }
  first_joint_vel_ref_received_ = true;
}

void DynamicController::setLinearKDCallback(
  const std::shared_ptr<highlevel_interfaces::srv::SetKD::Request> request,
  std::shared_ptr<highlevel_interfaces::srv::SetKD::Response> response)
{
  linear_k_ = request->k;
  linear_d_ = request->d;
  response->success = true;
}

void DynamicController::setJointKDCallback(
  const std::shared_ptr<highlevel_interfaces::srv::SetKD::Request> request,
  std::shared_ptr<highlevel_interfaces::srv::SetKD::Response> response)
{
  joint_k_ = request->k;
  joint_d_ = request->d;
  response->success = true;
}

Eigen::MatrixXd DynamicController::pseudoInverse(const Eigen::MatrixXd & matrix) const
{
  return matrix.completeOrthogonalDecomposition().pseudoInverse();
}

void DynamicController::computeKinematicsAndDynamics()
{
  // Update the robot model at the current q and qdot so we can compute task-space
  // feedback and all inverse-dynamics terms from the same state.
  pinocchio::forwardKinematics(model_, data_, joint_pos_, joint_vel_);
  pinocchio::updateFramePlacements(model_, data_);
  pinocchio::computeJointJacobians(model_, data_, joint_pos_);
  pinocchio::computeJointJacobiansTimeVariation(model_, data_, joint_pos_, joint_vel_);

  jacobian_.setZero();
  jacobian_dot_.setZero();
  pinocchio::getFrameJacobian(
    model_, data_, hand_frame_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian_);
  pinocchio::getFrameJacobianTimeVariation(
    model_, data_, hand_frame_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian_dot_);

  j_linear_ = jacobian_.topRows(3);
  jdot_linear_ = jacobian_dot_.topRows(3);

  const auto frame_pose = pinocchio::updateFramePlacement(model_, data_, hand_frame_id_);
  fbk_task_pos_ = frame_pose.translation();
  fbk_task_vel_ = pinocchio::getFrameVelocity(
    model_, data_, hand_frame_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED).linear();

  pinocchio::crba(model_, data_, joint_pos_);
  mass_matrix_ = data_.M;
  mass_matrix_.triangularView<Eigen::StrictlyLower>() =
    mass_matrix_.transpose().triangularView<Eigen::StrictlyLower>();
  nonlinear_effects_ = pinocchio::nonLinearEffects(model_, data_, joint_pos_, joint_vel_);
}

void DynamicController::publishFeedback()
{
  geometry_msgs::msg::Pose pose_msg;
  pose_msg.position.x = fbk_task_pos_[0];
  pose_msg.position.y = fbk_task_pos_[1];
  pose_msg.position.z = fbk_task_pos_[2];
  pose_msg.orientation.w = 1.0;
  pose_msg.orientation.x = 0.0;
  pose_msg.orientation.y = 0.0;
  pose_msg.orientation.z = 0.0;
  pose_pub_->publish(pose_msg);

  geometry_msgs::msg::Twist twist_msg;
  twist_msg.linear.x = fbk_task_vel_[0];
  twist_msg.linear.y = fbk_task_vel_[1];
  twist_msg.linear.z = fbk_task_vel_[2];
  twist_msg.angular.x = 0.0;
  twist_msg.angular.y = 0.0;
  twist_msg.angular.z = 0.0;
  twist_pub_->publish(twist_msg);
}

void DynamicController::computeAndPublishCommand()
{
  if (!first_pose_ref_received_ || !first_twist_ref_received_) {
    return;
  }

  const Eigen::Vector3d ref_task_acc = Eigen::Vector3d::Zero();
  const Eigen::Vector3d task_pos_error = ref_task_pos_ - fbk_task_pos_;
  const Eigen::Vector3d task_vel_error = ref_task_vel_ - fbk_task_vel_;
  // Task-space PD law from the slides: xddot_cmd = xddot_ref + D(xdot_ref - xdot) + K(x_ref - x).
  const Eigen::Vector3d xddot_cmd =
    ref_task_acc + linear_d_ * task_vel_error + linear_k_ * task_pos_error;

  const Eigen::MatrixXd mass_inv = mass_matrix_.ldlt().solve(
    Eigen::MatrixXd::Identity(dim_joints_, dim_joints_));
  // Task-space inertia: Lambda = (J M^-1 J^T)^-1.
  const Eigen::Matrix3d lambda_inv = j_linear_ * mass_inv * j_linear_.transpose();
  lambda_ = pseudoInverse(lambda_inv);

  const Eigen::MatrixXd j_bar = mass_inv * j_linear_.transpose() * lambda_;
  const Eigen::Vector3d jdot_qdot = jdot_linear_ * joint_vel_;
  // eta collects nonlinear dynamics and the Jdot*qdot correction term.
  const Eigen::Vector3d eta = j_bar.transpose() * nonlinear_effects_ - lambda_ * jdot_qdot;
  const Eigen::Vector3d wrench_cmd = lambda_ * xddot_cmd + eta;

  // Map the commanded task-space wrench back to joint torques.
  Eigen::VectorXd tau_cmd = j_linear_.transpose() * wrench_cmd;

  if (with_redundancy_ && first_joint_pos_ref_received_ && first_joint_vel_ref_received_) {
    // Null-space joint PD objective used only in the redundant version.
    const Eigen::VectorXd qddot_null_cmd =
      joint_d_ * (ref_joint_vel_ - joint_vel_) + joint_k_ * (ref_joint_pos_ - joint_pos_);
    const Eigen::VectorXd tau_joint = mass_matrix_ * qddot_null_cmd + nonlinear_effects_;
    // Dynamically consistent projector so null-space torques do not affect the 3D task.
    nullspace_projection_ =
      Eigen::MatrixXd::Identity(dim_joints_, dim_joints_) -
      j_linear_.transpose() * lambda_ * j_linear_ * mass_inv;
    tau_cmd += nullspace_projection_ * tau_joint;
  }

  cmd_joint_torque_ = tau_cmd;

  std_msgs::msg::Float64MultiArray torque_msg;
  torque_msg.data.resize(dim_joints_);
  for (int i = 0; i < dim_joints_; ++i) {
    torque_msg.data[i] = cmd_joint_torque_[i];
  }
  torque_pub_->publish(torque_msg);
}

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<DynamicController>());
  rclcpp::shutdown();
  return 0;
}
