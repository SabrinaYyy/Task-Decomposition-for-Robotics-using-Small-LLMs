#include "kinematic_controller/kinematic_controller.hpp"

#include <chrono>
#include <algorithm>

using namespace std::chrono_literals;

KinematicController::KinematicController()
: Node("kinematic_controller"),
  first_joint_received_(false),
  first_twist_received_(false),
  first_joint_ref_received_(false),
  publish_rate_(500.0),
  with_redundancy_(false)
{
  // Read parameters
  if (!readParameters()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to read parameters, shutting down.");
    rclcpp::shutdown();
    return;
  }

  // Initialize Pinocchio and internal buffers
  init();

  // QoS
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  // Subscribers
  joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", qos,
    std::bind(&KinematicController::jointStateCallback, this, std::placeholders::_1));

  twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "/gen3/reference/twist", qos,
    std::bind(&KinematicController::referenceTwistCallback, this, std::placeholders::_1));

  // Step V / redundancy: subscribe to joint planner output
  joint_ref_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/gen3/reference/velocity", qos,
    std::bind(&KinematicController::referenceJointVelCallback, this, std::placeholders::_1));

  // Publishers
  pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>("/gen3/feedback/pose", qos);
  twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/gen3/feedback/twist", qos);
  joint_cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/gen3/command/joint_velocity", qos);

  // Timer at publish_rate_ (500 Hz)
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / publish_rate_),
    std::bind(&KinematicController::publishFeedback, this));

  RCLCPP_INFO(this->get_logger(),
              "KinematicController initialized. rate=%.1f Hz with_redundancy=%s",
              publish_rate_, with_redundancy_ ? "true" : "false");
}

bool KinematicController::readParameters()
{
  this->declare_parameter<std::string>("urdf_file_name", "");
  this->declare_parameter<double>("publish_rate", 500.0);
  this->declare_parameter<bool>("with_redundancy", false);

  if (!this->get_parameter("urdf_file_name", urdf_file_name_)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to get urdf_file_name parameter");
    return false;
  }
  if (urdf_file_name_.empty()) {
    RCLCPP_ERROR(this->get_logger(), "urdf_file_name is empty");
    return false;
  }

  this->get_parameter("publish_rate", publish_rate_);
  this->get_parameter("with_redundancy", with_redundancy_);

  RCLCPP_INFO(this->get_logger(), "URDF file: %s", urdf_file_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "Publish rate: %.1f Hz", publish_rate_);
  RCLCPP_INFO(this->get_logger(), "With redundancy: %s", with_redundancy_ ? "true" : "false");
  return true;
}

void KinematicController::init()
{
  // Build Pinocchio model from URDF
  pinocchio::urdf::buildModel(urdf_file_name_, model_, false);
  data_ = pinocchio::Data(model_);

  // Same as prof example (bracelet_link)
  hand_id_ = model_.getJointId("bracelet_link") - 1;
  dim_joints_ = model_.nq;

  if (dim_joints_ <= 0) {
    RCLCPP_ERROR(this->get_logger(), "Invalid dim_joints_=%d", dim_joints_);
  }

  RCLCPP_INFO(this->get_logger(), "model.nq=%d model.nv=%d", model_.nq, model_.nv);
  RCLCPP_INFO(this->get_logger(), "dim_joints_=%d hand_id_=%d", dim_joints_, hand_id_);

  // State
  joint_pos_ = Eigen::VectorXd::Zero(dim_joints_);
  joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);
  fbk_task_pos_ = Eigen::Vector3d::Zero();
  fbk_task_vel_ = Eigen::Vector3d::Zero();

  // IK matrices
  jacobian_ = Eigen::MatrixXd::Zero(6, dim_joints_);
  J_linear_ = Eigen::MatrixXd::Zero(3, dim_joints_);
  J_pinv_ = Eigen::MatrixXd::Zero(dim_joints_, 3);

  // Redundancy
  N_ = Eigen::MatrixXd::Identity(dim_joints_, dim_joints_);
  ref_joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);

  // References / commands
  ref_task_vel_ = Eigen::Vector3d::Zero();
  cmd_joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);
}

void KinematicController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  // Copy joint positions (and velocities if available)
  const size_t npos = std::min(msg->position.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < npos; ++i) {
    joint_pos_[static_cast<int>(i)] = msg->position[i];
  }

  const size_t nvel = std::min(msg->velocity.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < nvel; ++i) {
    joint_vel_[static_cast<int>(i)] = msg->velocity[i];
  }
  // If velocity not provided, leave remaining entries as-is; or force to zero:
  if (msg->velocity.size() < npos) {
    for (size_t i = nvel; i < npos; ++i) joint_vel_[static_cast<int>(i)] = 0.0;
  }

  if (!first_joint_received_) {
    first_joint_received_ = true;
    RCLCPP_INFO(this->get_logger(), "First /joint_states received.");
  }
}

void KinematicController::referenceTwistCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
  // Translation-only task velocity (x_dot)
  ref_task_vel_[0] = msg->linear.x;
  ref_task_vel_[1] = msg->linear.y;
  ref_task_vel_[2] = msg->linear.z;

  if (!first_twist_received_) {
    first_twist_received_ = true;
    RCLCPP_INFO(this->get_logger(), "First /gen3/reference/twist received.");
  }
}

void KinematicController::referenceJointVelCallback(const std_msgs::msg::Float64MultiArray::SharedPtr msg)
{
  const size_t n = std::min(msg->data.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < n; ++i) {
    ref_joint_vel_[static_cast<int>(i)] = msg->data[i];
  }
  // If shorter than expected, zero remainder
  for (size_t i = n; i < static_cast<size_t>(dim_joints_); ++i) {
    ref_joint_vel_[static_cast<int>(i)] = 0.0;
  }

  if (!first_joint_ref_received_) {
    first_joint_ref_received_ = true;
    RCLCPP_INFO(this->get_logger(), "First /gen3/reference/velocity received.");
  }
}

void KinematicController::computeForwardKinematics()
{
  // Compute forward kinematics with current q, dq
  pinocchio::forwardKinematics(model_, data_, joint_pos_, joint_vel_);

  // End-effector pose (translation only)
  const pinocchio::SE3 pose_now = data_.oMi[hand_id_];
  fbk_task_pos_ = pose_now.translation();

  // A simple “consistent” EE linear velocity estimate:
  // (Pinocchio velocity access can vary; this keeps it stable for feedback)
  // If you want, you can replace with getJointVelocity in LOCAL_WORLD_ALIGNED.
  fbk_task_vel_ = data_.v[hand_id_].linear();
}

void KinematicController::computeJacobian()
{
  // Compute Jacobian in LOCAL_WORLD_ALIGNED like the example
  // Using computeAllTerms is also fine; but for just Jacobian, these are enough:
  pinocchio::computeAllTerms(model_, data_, joint_pos_, joint_vel_);
  pinocchio::getJointJacobian(
    model_, data_, hand_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian_);

  // translation-only Jacobian: 3x7
  J_linear_ = jacobian_.topRows(3);

  // pseudo-inverse: 7x3
  J_pinv_ = J_linear_.completeOrthogonalDecomposition().pseudoInverse();
}

void KinematicController::computeInverseKinematics()
{
  // Don’t compute/publish command until you have both joint feedback and reference twist
  if (!first_joint_received_ || !first_twist_received_) return;

  // Basic IK: qdot = J^+ * xdot_ref
  cmd_joint_vel_ = J_pinv_ * ref_task_vel_;

  // Redundancy term: qdot += N * qdot_ref
  if (with_redundancy_ && first_joint_ref_received_) {
    N_ = Eigen::MatrixXd::Identity(dim_joints_, dim_joints_) - (J_pinv_ * J_linear_);
    cmd_joint_vel_ += N_ * ref_joint_vel_;
  }

  // Publish joint velocity command
  std_msgs::msg::Float64MultiArray cmd_msg;
  cmd_msg.data.resize(dim_joints_);
  for (int i = 0; i < dim_joints_; ++i) {
    cmd_msg.data[i] = cmd_joint_vel_[i];
  }
  joint_cmd_pub_->publish(cmd_msg);
}

void KinematicController::publishFeedback()
{
  // Fixed-rate loop does EVERYTHING: FK + Jacobian + publish feedback + publish command
  if (!first_joint_received_) return;

  computeForwardKinematics();
  computeJacobian();

  // Publish pose feedback
  geometry_msgs::msg::Pose pose_msg;
  pose_msg.position.x = fbk_task_pos_[0];
  pose_msg.position.y = fbk_task_pos_[1];
  pose_msg.position.z = fbk_task_pos_[2];
  // Orientation not controlled in this assignment; publish identity
  pose_msg.orientation.w = 1.0;
  pose_msg.orientation.x = 0.0;
  pose_msg.orientation.y = 0.0;
  pose_msg.orientation.z = 0.0;
  pose_pub_->publish(pose_msg);

  // Publish twist feedback (linear only)
  geometry_msgs::msg::Twist twist_msg;
  twist_msg.linear.x = fbk_task_vel_[0];
  twist_msg.linear.y = fbk_task_vel_[1];
  twist_msg.linear.z = fbk_task_vel_[2];
  twist_msg.angular.x = 0.0;
  twist_msg.angular.y = 0.0;
  twist_msg.angular.z = 0.0;
  twist_pub_->publish(twist_msg);

  // Compute/publish command at the same fixed rate
  computeInverseKinematics();
}

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KinematicController>());
  rclcpp::shutdown();
  return 0;
}
