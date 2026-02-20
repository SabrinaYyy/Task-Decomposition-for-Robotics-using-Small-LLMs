#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <highlevel_interfaces/srv/move3d.hpp>
#include <Eigen/Dense>
#include <cmath>

class PotentialFieldTask : public rclcpp::Node {
public:
    PotentialFieldTask() : Node("potential_field_task"), 
                           target_set_(false),
                           first_feedback_received_(false) {
        // Read parameters
        if (!readParameters()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to read parameters");
            rclcpp::shutdown();
            return;
        }
        
        // Create service
        service_ = this->create_service<highlevel_interfaces::srv::Move3d>(
            "/planner/move_to",
            std::bind(&PotentialFieldTask::serviceCallback, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        // Create subscriber to feedback pose
        pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/gen3/feedback/pose", 10,
            std::bind(&PotentialFieldTask::poseCallback, this, std::placeholders::_1));
        
        // Create publisher for reference twist
        twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/gen3/reference/twist", 10);
        
        // Create timer for publishing at fixed rate
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_rate_),
            std::bind(&PotentialFieldTask::timerCallback, this));
        
        RCLCPP_INFO(this->get_logger(), "Potential Field Task initialized");
    }

private:
    bool readParameters() {
        this->declare_parameter<double>("publish_rate", 500.0);
        this->declare_parameter<double>("k_att_linear", 5.0);
        this->declare_parameter<double>("maximum_linear_velocity", 1.0);
        this->declare_parameter<double>("done_threshold", 1e-4);
        
        this->get_parameter("publish_rate", publish_rate_);
        this->get_parameter("k_att_linear", k_att_linear_);
        this->get_parameter("maximum_linear_velocity", max_linear_velocity_);
        this->get_parameter("done_threshold", done_threshold_);

        RCLCPP_INFO(this->get_logger(), "Publish rate: %.1f Hz", publish_rate_);
        RCLCPP_INFO(this->get_logger(), "K_att_linear: %.2f", k_att_linear_);
        RCLCPP_INFO(this->get_logger(), "Max linear velocity: %.2f m/s", max_linear_velocity_);
        RCLCPP_INFO(this->get_logger(), "Done threshold: %.6f m", done_threshold_);

        return true;
    }
    
    void serviceCallback(
        const std::shared_ptr<highlevel_interfaces::srv::Move3d::Request> request,
        std::shared_ptr<highlevel_interfaces::srv::Move3d::Response> response) {
        
        target_pos_[0] = request->x;
        target_pos_[1] = request->y;
        target_pos_[2] = request->z;
        target_set_ = true;
        
        RCLCPP_INFO(this->get_logger(), "New target: [%.3f, %.3f, %.3f]",
                    target_pos_[0], target_pos_[1], target_pos_[2]);
        
        response->success = true;
    }
    
    void poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg) {
        current_pos_[0] = msg->position.x;
        current_pos_[1] = msg->position.y;
        current_pos_[2] = msg->position.z;
        
        if (!first_feedback_received_) {
            first_feedback_received_ = true;
            RCLCPP_INFO(this->get_logger(), "First pose feedback received");
        }
    }
    
    void timerCallback() {
        // Don't publish if we haven't received feedback or target
        if (!first_feedback_received_ || !target_set_) {
            return;
        }
        
        // Compute potential field
        Eigen::Vector3d error = target_pos_ - current_pos_;
        Eigen::Vector3d reference_vel = k_att_linear_ * error;
        
        // Saturate velocity
        double vel_norm = reference_vel.norm();
        if (vel_norm > max_linear_velocity_) {
            reference_vel = reference_vel * (max_linear_velocity_ / vel_norm);
        }
        
        // Check if reached target (within 1cm)
        if (error.norm() < done_threshold_) {
            reference_vel.setZero();
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "Target reached!");
        }
        
        // Publish reference twist
        geometry_msgs::msg::Twist twist_msg;
        twist_msg.linear.x = reference_vel[0];
        twist_msg.linear.y = reference_vel[1];
        twist_msg.linear.z = reference_vel[2];
        twist_msg.angular.x = 0.0;
        twist_msg.angular.y = 0.0;
        twist_msg.angular.z = 0.0;
        
        twist_pub_->publish(twist_msg);
    }
    
    // ROS interfaces
    rclcpp::Service<highlevel_interfaces::srv::Move3d>::SharedPtr service_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    // Parameters
    double publish_rate_;
    double k_att_linear_;
    double max_linear_velocity_;
    double done_threshold_;

    // State
    Eigen::Vector3d current_pos_;
    Eigen::Vector3d target_pos_;
    bool target_set_;
    bool first_feedback_received_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PotentialFieldTask>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}