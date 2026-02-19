#ifndef KINEMATIC_CONTROLLER_HPP
#define KINEMATIC_CONTROLLER_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include <pinocchio/fwd.hpp>
#include <pinocchio/multibody/model.hpp>
#include <pinocchio/multibody/data.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <Eigen/Dense>

class KinematicController : public rclcpp::Node {
public:
    KinematicController();

private:
    // Initialization
    void init();
    bool readParameters();
    
    // Callbacks
    void jointStateCallback(const sensor_msgs::msg::JointState::SharedPtr msg);
    
    // Computation
    void computeForwardKinematics();
    void publishFeedback();
    
    // Pinocchio objects
    pinocchio::Model model_;
    pinocchio::Data data_;
    int hand_id_;
    int dim_joints_;
    
    // State variables
    Eigen::VectorXd joint_pos_;
    Eigen::VectorXd joint_vel_;
    Eigen::Vector3d fbk_task_pos_;
    Eigen::Vector3d fbk_task_vel_;
    
    // Flags
    bool first_joint_received_;
    
    // ROS interfaces
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Parameters
    std::string urdf_file_name_;
    double publish_rate_;
};

#endif