#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <highlevel_interfaces/srv/move3d.hpp>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

class PotentialFieldTask : public rclcpp::Node {
public:
    PotentialFieldTask() : Node("potential_field_task"), 
                           first_feedback_received_(false) {
        //read params
        if (!readParameters()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to read parameters");
            rclcpp::shutdown();
            return;
        }
        
        //create service
        service_ = this->create_service<highlevel_interfaces::srv::Move3d>(
            "/planner/move_to",
            std::bind(&PotentialFieldTask::serviceCallback, this,
                     std::placeholders::_1, std::placeholders::_2));
        
        //create subscriber to feedback pose
        pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
            "/gen3/feedback/pose", 10,
            std::bind(&PotentialFieldTask::poseCallback, this, std::placeholders::_1));
        
        //create publisher for reference pose and twist
        pose_pub_ = this->create_publisher<geometry_msgs::msg::Pose>(
            "/gen3/reference/pose", 10);
        twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
            "/gen3/reference/twist", 10);
        
        //create timer for publishing at fixed rate
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
        this->declare_parameter<std::vector<double>>(
            "default_target_position", std::vector<double>{0.5, 0.2, 0.5});
        
        this->get_parameter("publish_rate", publish_rate_);
        this->get_parameter("k_att_linear", k_att_linear_);
        this->get_parameter("maximum_linear_velocity", max_linear_velocity_);
        this->get_parameter("done_threshold", done_threshold_);
        const auto default_target = this->get_parameter("default_target_position").as_double_array();
        if (default_target.size() == 3) {
            target_pos_[0] = default_target[0];
            target_pos_[1] = default_target[1];
            target_pos_[2] = default_target[2];
        } else {
            target_pos_ = Eigen::Vector3d(0.5, 0.2, 0.5);
        }

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
        if (!first_feedback_received_) {
            return;
        }
        
        //compute potential field
        Eigen::Vector3d error = target_pos_ - current_pos_;
        Eigen::Vector3d reference_vel = k_att_linear_ * error;
        
        //saturate velocity
        double vel_norm = reference_vel.norm();
        if (vel_norm > max_linear_velocity_) {
            reference_vel = reference_vel * (max_linear_velocity_ / vel_norm);
        }
        
        //check if reached target (within 1mm)
        if (error.norm() < done_threshold_) {
            reference_vel.setZero();
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                               "Target reached!");
        }

        const double dt = 1.0 / publish_rate_;
        const Eigen::Vector3d reference_pos = current_pos_ + dt * reference_vel;

        geometry_msgs::msg::Pose pose_msg;
        pose_msg.position.x = reference_pos[0];
        pose_msg.position.y = reference_pos[1];
        pose_msg.position.z = reference_pos[2];
        pose_msg.orientation.w = 1.0;
        pose_msg.orientation.x = 0.0;
        pose_msg.orientation.y = 0.0;
        pose_msg.orientation.z = 0.0;
        pose_pub_->publish(pose_msg);
        
        //publish reference twist
        geometry_msgs::msg::Twist twist_msg;
        twist_msg.linear.x = reference_vel[0];
        twist_msg.linear.y = reference_vel[1];
        twist_msg.linear.z = reference_vel[2];
        twist_msg.angular.x = 0.0;
        twist_msg.angular.y = 0.0;
        twist_msg.angular.z = 0.0;
        
        twist_pub_->publish(twist_msg);
    }
    
    //rosinterfaces
    rclcpp::Service<highlevel_interfaces::srv::Move3d>::SharedPtr service_;
    rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Pose>::SharedPtr pose_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    //parameters
    double publish_rate_;
    double k_att_linear_;
    double max_linear_velocity_;
    double done_threshold_;

    //state
    Eigen::Vector3d current_pos_;
    Eigen::Vector3d target_pos_;
    bool first_feedback_received_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<PotentialFieldTask>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
