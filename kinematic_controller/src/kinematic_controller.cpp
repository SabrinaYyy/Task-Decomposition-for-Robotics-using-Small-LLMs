#include "kinematic_controller/kinematic_controller.hpp"

KinematicController::KinematicController() 
    : Node("kinematic_controller"), first_joint_received_(false) {
    
    // Read parameters
    if (!readParameters()) {
        RCLCPP_ERROR(this->get_logger(), "Failed to read parameters");
        rclcpp::shutdown();
        return;
    }
    
    // Initialize Pinocchio
    init();
    
    // Create subscribers
    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
        "/joint_states", 10,
        std::bind(&KinematicController::jointStateCallback, this, std::placeholders::_1));
    
    // Create publishers
    pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>(
        "/gen3/feedback/pose", 10);
    twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
        "/gen3/feedback/twist", 10);
    
    // Create timer for publishing at fixed rate
    timer_ = this->create_wall_timer(
        std::chrono::duration<double>(1.0 / publish_rate_),
        std::bind(&KinematicController::publishFeedback, this));
    
    RCLCPP_INFO(this->get_logger(), "Kinematic Controller initialized");
}

bool KinematicController::readParameters() {
    // Declare and get parameters
    this->declare_parameter<std::string>("urdf_file_name", "");
    this->declare_parameter<double>("publish_rate", 500.0);
    
    if (!this->get_parameter("urdf_file_name", urdf_file_name_)) {
        RCLCPP_ERROR(this->get_logger(), "Failed to get urdf_file_name parameter");
        return false;
    }
    
    if (urdf_file_name_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "urdf_file_name is empty");
        return false;
    }
    
    this->get_parameter("publish_rate", publish_rate_);
    
    RCLCPP_INFO(this->get_logger(), "URDF file: %s", urdf_file_name_.c_str());
    RCLCPP_INFO(this->get_logger(), "Publish rate: %.1f Hz", publish_rate_);
    
    return true;
}

void KinematicController::init() {
    // Build Pinocchio model from URDF
    pinocchio::urdf::buildModel(urdf_file_name_, model_, false);
    data_ = pinocchio::Data(model_);
    
    // ====== ADD THIS DEBUG CODE HERE ======
    RCLCPP_INFO(this->get_logger(), "=== Available frames in model ===");
    for (size_t i = 0; i < model_.frames.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "Frame %zu: %s", i, model_.frames[i].name.c_str());
    }
    RCLCPP_INFO(this->get_logger(), "=== Available joints in model ===");
    for (size_t i = 0; i < model_.joints.size(); ++i) {
        RCLCPP_INFO(this->get_logger(), "Joint %zu: %s", i, model_.names[i].c_str());
    }
    // ====== END DEBUG CODE ======
    
    // Get end-effector joint ID (bracelet_link for Gen3)
    hand_id_ = model_.getJointId("bracelet_link") - 1;
    dim_joints_ = model_.nq;
    
    RCLCPP_INFO(this->get_logger(), "Number of joints: %d", dim_joints_);
    RCLCPP_INFO(this->get_logger(), "Hand ID: %d", hand_id_);
    
    // Initialize vectors
    joint_pos_ = Eigen::VectorXd::Zero(dim_joints_);
    joint_vel_ = Eigen::VectorXd::Zero(dim_joints_);
    fbk_task_pos_ = Eigen::Vector3d::Zero();
    fbk_task_vel_ = Eigen::Vector3d::Zero();
}

void KinematicController::jointStateCallback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
    
    // Copy joint positions and velocities
    for (size_t i = 0; i < msg->position.size() && i < (size_t)dim_joints_; ++i) {
        joint_pos_[i] = msg->position[i];
        joint_vel_[i] = msg->velocity[i];
    }
    
    // Compute forward kinematics
    computeForwardKinematics();
    
    if (!first_joint_received_) {
        first_joint_received_ = true;
        RCLCPP_INFO(this->get_logger(), "First joint state received");
    }
}

void KinematicController::computeForwardKinematics() {
    // Compute forward kinematics
    pinocchio::forwardKinematics(model_, data_, joint_pos_, joint_vel_);
    
    // Get end-effector position (translation only)
    fbk_task_pos_ = data_.oMi[hand_id_].translation();
    
    // Get end-effector velocity (linear part of spatial velocity)
    // v = J * q_dot (we'll compute this properly later with Jacobian)
    // For now, approximate with velocity of the joint
    fbk_task_vel_ = data_.v[hand_id_].linear();
}

void KinematicController::publishFeedback() {
    // Don't publish before receiving first joint state
    if (!first_joint_received_) {
        return;
    }
    
    // Publish pose
    geometry_msgs::msg::Pose pose_msg;
    pose_msg.position.x = fbk_task_pos_[0];
    pose_msg.position.y = fbk_task_pos_[1];
    pose_msg.position.z = fbk_task_pos_[2];
    // Orientation: leave as default (identity quaternion)
    pose_msg.orientation.w = 1.0;
    pose_msg.orientation.x = 0.0;
    pose_msg.orientation.y = 0.0;
    pose_msg.orientation.z = 0.0;
    
    pose_pub_->publish(pose_msg);
    
    // Publish twist
    geometry_msgs::msg::Twist twist_msg;
    twist_msg.linear.x = fbk_task_vel_[0];
    twist_msg.linear.y = fbk_task_vel_[1];
    twist_msg.linear.z = fbk_task_vel_[2];
    // Angular: leave as zero for now
    twist_msg.angular.x = 0.0;
    twist_msg.angular.y = 0.0;
    twist_msg.angular.z = 0.0;
    
    twist_pub_->publish(twist_msg);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<KinematicController>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}