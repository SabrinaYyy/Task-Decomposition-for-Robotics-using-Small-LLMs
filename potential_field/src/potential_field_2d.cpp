#include <cmath>
#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "highlevel_interfaces/srv/move2d.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

class PotentialField2D : public rclcpp::Node
{
public:
  PotentialField2D() : Node("potential_field_2d")
  {
    // Subscriber: robot pose feedback
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
      "/husky/platform/pose", rclcpp::QoS(10),
      std::bind(&PotentialField2D::pose_callback, this, _1));

    // Publishers expected by the controller
    vel_pub_  = this->create_publisher<std_msgs::msg::Float64MultiArray>("/planner/velocity", rclcpp::QoS(10));
    done_pub_ = this->create_publisher<std_msgs::msg::Bool>("/planner/done", rclcpp::QoS(10));

    // Service server: user provides target (x,y)
    move_srv_ = this->create_service<highlevel_interfaces::srv::Move2d>(
      "/planner/move_to",
      std::bind(&PotentialField2D::service_callback, this, _1, _2));

    // 500 Hz = 2 ms
    using namespace std::chrono_literals;
    timer_ = this->create_wall_timer(2ms, std::bind(&PotentialField2D::update, this));

    RCLCPP_INFO(this->get_logger(), "potential_field_2d started (500 Hz).");
  }

private:
  void pose_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (msg->poses.size() <= 5) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "PoseArray size=%zu; expected >= 6 (need poses[5]).", msg->poses.size());
      return;
    }

    cur_x_ = msg->poses[5].position.x;
    cur_y_ = msg->poses[5].position.y;
    got_pose_ = true;

    RCLCPP_INFO_ONCE(this->get_logger(), "Received first pose: (%.3f, %.3f)", cur_x_, cur_y_);
  }

  void service_callback(
    const std::shared_ptr<highlevel_interfaces::srv::Move2d::Request> request,
    std::shared_ptr<highlevel_interfaces::srv::Move2d::Response> response)
  {
    // Return false if service request is received before first pose feedback
    if (!got_pose_) {
      response->success = false;
      RCLCPP_WARN(this->get_logger(), "Service called before first pose; success=false.");
      return;
    }

    target_x_ = request->x;
    target_y_ = request->y;
    got_target_ = true;

    response->success = true;
    RCLCPP_INFO(this->get_logger(), "Target set to (%.3f, %.3f).", target_x_, target_y_);
  }

  void update()
  {
    std_msgs::msg::Float64MultiArray vel_msg;
    std_msgs::msg::Bool done_msg;

    // Always publish 2 values: [vx, vy]
    vel_msg.data.resize(2);

    // Before first pose feedback OR before first service call:
    // publish (0,0) and done=true
    if (!got_pose_ || !got_target_) {
      vel_msg.data[0] = 0.0;
      vel_msg.data[1] = 0.0;
      done_msg.data = true;

      vel_pub_->publish(vel_msg);
      done_pub_->publish(done_msg);
      return;
    }

    constexpr double katt = 5.0;
    constexpr double vmax = 1.0;

    const double dx = target_x_ - cur_x_;
    const double dy = target_y_ - cur_y_;
    const double dist = std::sqrt(dx * dx + dy * dy);

    // Done condition
    done_msg.data = (dist < 0.1);

    // Potential field velocity (attractive)
    double vx = katt * dx;
    double vy = katt * dy;

    // Normalize only if norm > vmax
    const double vnorm = std::sqrt(vx * vx + vy * vy);
    if (vnorm > vmax && vnorm > 1e-9) {
      vx = (vx / vnorm) * vmax;
      vy = (vy / vnorm) * vmax;
    }

    // Publish velocity (even when done=true, per prof/autograder comment)
    vel_msg.data[0] = vx;
    vel_msg.data[1] = vy;

    vel_pub_->publish(vel_msg);
    done_pub_->publish(done_msg);
  }

  // ROS interfaces
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
  rclcpp::Service<highlevel_interfaces::srv::Move2d>::SharedPtr move_srv_;
  rclcpp::TimerBase::SharedPtr timer_;

  // State
  bool got_pose_{false};
  bool got_target_{false};

  double cur_x_{0.0};
  double cur_y_{0.0};

  double target_x_{0.0};
  double target_y_{0.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PotentialField2D>());
  rclcpp::shutdown();
  return 0;
}
