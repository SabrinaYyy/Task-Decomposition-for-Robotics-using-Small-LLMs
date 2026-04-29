#include "kinematic_controller/kinematic_controller.hpp"

#include <algorithm>
#include <chrono>

using namespace std::chrono_literals;

// I use this node as the task-space velocity controller for the Gen3. It reads
// the arm state, computes FK/Jacobians with Pinocchio, and converts the desired
// end-effector twist into joint velocity commands.
KinematicController::KinematicController()
: Node("kinematic_controller"),
  //with_redundancy_(false),
  first_joint_received_(false),
  first_twist_received_(false),
  first_joint_ref_received_(false)
  
{
  // I read these from launch files so I can switch between 3D and 6D control.
  if (!readParameters()) {
    RCLCPP_ERROR(this->get_logger(), "Failed to read parameters, shutting down.");
    rclcpp::shutdown();
    return;
  }

  // I build the Pinocchio model once and reuse the buffers every control tick.
  init();

  // QoS
  auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

  // /joint_states is the measured robot state from Gazebo/ros2_control.
  joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", qos,
    std::bind(&KinematicController::jointStateCallback, this, std::placeholders::_1));

  twist_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
    "/gen3/reference/twist", qos,
    std::bind(&KinematicController::referenceTwistCallback, this, std::placeholders::_1));

  // I keep this subscriber for redundancy experiments, but the final demo runs
  // with redundancy disabled because it was more stable.
  joint_ref_sub_ = this->create_subscription<std_msgs::msg::Float64MultiArray>(
    "/gen3/reference/velocity", qos,
    std::bind(&KinematicController::referenceJointVelCallback, this, std::placeholders::_1));

  // I publish feedback for the planner and the final joint velocity command for
  // the low-level controller bridge.
  pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>("/gen3/feedback/pose", qos);
  twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/gen3/feedback/twist", qos);
  joint_cmd_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    "/gen3/command/joint_velocity", qos);

  // The whole feedback/IK loop runs at the same rate as the controller.
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / publish_rate_),
    std::bind(&KinematicController::publishFeedback, this));

  RCLCPP_INFO(this->get_logger(),
              "KinematicController initialized. rate=%.1f Hz with_redundancy=%s control_orientation=%s",
              publish_rate_,
              with_redundancy_ ? "true" : "false",
              control_orientation_ ? "true" : "false");
}

bool KinematicController::readParameters()
{
  this->declare_parameter<std::string>("urdf_file_name", "");
  this->declare_parameter<double>("publish_rate", 500.0);
  this->declare_parameter<bool>("with_redundancy", false);
  this->declare_parameter<bool>("control_orientation", false);

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
  this->get_parameter("control_orientation", control_orientation_);

  RCLCPP_INFO(this->get_logger(), "URDF file: %s", urdf_file_name_.c_str());
  RCLCPP_INFO(this->get_logger(), "Publish rate: %.1f Hz", publish_rate_);
  RCLCPP_INFO(this->get_logger(), "With redundancy: %s", with_redundancy_ ? "true" : "false");
  RCLCPP_INFO(this->get_logger(), "Control orientation: %s", control_orientation_ ? "true" : "false");
  return true;
}

void KinematicController::init()
{
  // I build the kinematic model from the same URDF used in simulation.
  pinocchio::urdf::buildModel(urdf_file_name_, model_, false);
  data_ = pinocchio::Data(model_);

  // I control bracelet_link as the end-effector because that is the link used
  // throughout my previous task-space assignments.
  hand_id_ = model_.getJointId("bracelet_link") - 1;
  dim_joints_ = model_.nq;

  if (dim_joints_ <= 0) {
    RCLCPP_ERROR(this->get_logger(), "Invalid dim_joints_=%d", dim_joints_);
  }

  RCLCPP_INFO(this->get_logger(), "model.nq=%d model.nv=%d", model_.nq, model_.nv);
  RCLCPP_INFO(this->get_logger(), "dim_joints_=%d hand_id_=%d", dim_joints_, hand_id_);

  // I initialize all state vectors before the first /joint_states message arrives.
  joint_pos_ = Eigen::VectorXd::Zero(dim_joints_);
  joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);
  fbk_task_pos_ = Eigen::Vector3d::Zero();
  fbk_task_orientation_ = Eigen::Quaterniond::Identity();
  fbk_task_vel_ = Eigen::Vector3d::Zero();
  fbk_task_angular_vel_ = Eigen::Vector3d::Zero();

  // I keep both 3D and 6D Jacobian paths so the launch file can choose the mode.
  jacobian_ = Eigen::MatrixXd::Zero(6, dim_joints_);
  J_linear_ = Eigen::MatrixXd::Zero(3, dim_joints_);
  J_task_ = Eigen::MatrixXd::Zero(6, dim_joints_);
  J_linear_pinv_ = Eigen::MatrixXd::Zero(dim_joints_, 3);
  J_task_pinv_ = Eigen::MatrixXd::Zero(dim_joints_, 6);
  J_active_ = Eigen::MatrixXd::Zero(3, dim_joints_);
  J_active_pinv_ = Eigen::MatrixXd::Zero(dim_joints_, 3);

  // Redundancy state is only used when with_redundancy=true.
  N_ = Eigen::MatrixXd::Identity(dim_joints_, dim_joints_);
  ref_joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);

  // These hold the latest task-space command and computed joint command.
  ref_task_vel_.setZero();
  cmd_joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);
}

void KinematicController::jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  // I copy only the first seven joints because the gripper appears as an extra
  // joint in /joint_states and should not enter the arm Jacobian.
  const size_t npos = std::min(msg->position.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < npos; ++i) {
    joint_pos_[static_cast<int>(i)] = msg->position[i];
  }

  const size_t nvel = std::min(msg->velocity.size(), static_cast<size_t>(dim_joints_));
  for (size_t i = 0; i < nvel; ++i) {
    joint_vel_[static_cast<int>(i)] = msg->velocity[i];
  }
  // Gazebo does not always publish velocity; I zero missing entries so I do not
  // reuse stale values.
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
  // I store the planner output as a 6D task-space command:
  // [vx vy vz wx wy wz]^T.
  ref_task_vel_[0] = msg->linear.x;
  ref_task_vel_[1] = msg->linear.y;
  ref_task_vel_[2] = msg->linear.z;
  ref_task_vel_[3] = msg->angular.x;
  ref_task_vel_[4] = msg->angular.y;
  ref_task_vel_[5] = msg->angular.z;

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
  // If the message is shorter than expected, I zero the rest to avoid stale commands.
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
  // I update FK from the current joint state so the planner can see where the
  // end-effector is right now.
  pinocchio::forwardKinematics(model_, data_, joint_pos_, joint_vel_);

  // I publish both position and orientation even though the final demo uses
  // position-only control.
  const pinocchio::SE3 pose_now = data_.oMi[hand_id_];
  fbk_task_pos_ = pose_now.translation();
  fbk_task_orientation_ = Eigen::Quaterniond(pose_now.rotation());
  fbk_task_orientation_.normalize();
}

void KinematicController::computeJacobian()
{
  // I build the geometric Jacobian relating qdot to end-effector twist:
  // xdot = J(q) qdot.
  pinocchio::computeJointJacobians(model_, data_, joint_pos_);
  pinocchio::getJointJacobian(
    model_, data_, hand_id_, pinocchio::ReferenceFrame::LOCAL_WORLD_ALIGNED, jacobian_);

  // In 3D mode I only use the translational rows; in 6D mode I use all rows.
  J_linear_ = jacobian_.topRows(3);
  J_task_ = jacobian_;

  // The arm is redundant, so I use a pseudoinverse for the IK solve.
  J_linear_pinv_ = J_linear_.completeOrthogonalDecomposition().pseudoInverse();
  J_task_pinv_ = J_task_.completeOrthogonalDecomposition().pseudoInverse();

  if (control_orientation_) {
    // In 6D mode I control translation and orientation together.
    J_active_ = J_task_;
    J_active_pinv_ = J_task_pinv_;
  } else {
    J_active_ = J_linear_;
    J_active_pinv_ = J_linear_pinv_;
  }

}

void KinematicController::computeInverseKinematics()
{
  // I wait until both feedback and a reference twist exist before commanding motion.
  if (!first_joint_received_ || !first_twist_received_) return;

  if (control_orientation_) {
    // Full 6D mode is available, but I do not use it in the final demo.
    cmd_joint_vel_ = J_active_pinv_ * ref_task_vel_;
  } else {
    // Translation-only IK is the stable mode used for the option 4 tasks.
    cmd_joint_vel_ = J_active_pinv_ * ref_task_vel_.head<3>();
  }

  // Redundancy resolution adds posture motion in the Jacobian null space:
  // qdot = J^dagger xdot_ref + (I - J^dagger J) qdot_ref.
  if (with_redundancy_ && first_joint_ref_received_) {
    N_ = Eigen::MatrixXd::Identity(dim_joints_, dim_joints_) - (J_active_pinv_ * J_active_);
    cmd_joint_vel_ += N_ * ref_joint_vel_;
  }

  // I publish joint velocities; the Kortex side converts these into position
  // commands for the active controller.
  std_msgs::msg::Float64MultiArray cmd_msg;
  cmd_msg.data.resize(dim_joints_);
  for (int i = 0; i < dim_joints_; ++i) {
    cmd_msg.data[i] = cmd_joint_vel_[i];
  }
  joint_cmd_pub_->publish(cmd_msg);
}

void KinematicController::publishFeedback()
{
  // This timer is my fixed-rate pipeline:
  // joint feedback -> FK/Jacobian -> task feedback -> IK command.
  if (!first_joint_received_) return;

  computeForwardKinematics();
  computeJacobian();
  // I recover feedback twist from xdot = J qdot.
  fbk_task_vel_ = J_linear_ * joint_vel_;
  fbk_task_angular_vel_ = jacobian_.bottomRows(3) * joint_vel_;

  // I publish this pose so the potential-field action server can close the loop.
  geometry_msgs::msg::Pose pose_msg;
  pose_msg.position.x = fbk_task_pos_[0];
  pose_msg.position.y = fbk_task_pos_[1];
  pose_msg.position.z = fbk_task_pos_[2];
  pose_msg.orientation.x = fbk_task_orientation_.x();
  pose_msg.orientation.y = fbk_task_orientation_.y();
  pose_msg.orientation.z = fbk_task_orientation_.z();
  pose_msg.orientation.w = fbk_task_orientation_.w();
  pose_pub_->publish(pose_msg);

  // Publish twist feedback
  // The feedback twist contains both linear and angular velocity.
  geometry_msgs::msg::Twist twist_msg;
  twist_msg.linear.x = fbk_task_vel_[0];
  twist_msg.linear.y = fbk_task_vel_[1];
  twist_msg.linear.z = fbk_task_vel_[2];
  twist_msg.angular.x = fbk_task_angular_vel_[0];
  twist_msg.angular.y = fbk_task_angular_vel_[1];
  twist_msg.angular.z = fbk_task_angular_vel_[2];
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
