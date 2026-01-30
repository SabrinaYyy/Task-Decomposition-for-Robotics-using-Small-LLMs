#include <cmath>
#include <chrono>
#include <memory>

#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/pose_array.hpp" //Robot pose message
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp" //Velocity command
#include "highlevel_interfaces/srv/move2d.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

class PotentialField2D : public rclcpp::Node
{
public:
  PotentialField2D() : Node("potential_field_2d")
  {
    //subscribe to robot's current pose (position/orientation array)
    pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseArray>(
      "/husky/platform/pose", //topic name
      rclcpp::QoS(10), //queue size of 10 messages
      std::bind(&PotentialField2D::pose_callback, this, _1));

    //publish velocity commands [vx, vy]
    vel_pub_  = this->create_publisher<std_msgs::msg::Float64MultiArray>("/planner/velocity", rclcpp::QoS(10));
    //publish completion status
    done_pub_ = this->create_publisher<std_msgs::msg::Bool>("/planner/done", rclcpp::QoS(10));

    //service to receive target (x,y)
    move_srv_ = this->create_service<highlevel_interfaces::srv::Move2d>(
      "/planner/move_to",
      std::bind(&PotentialField2D::service_callback, this, _1, _2));

    //500 Hz = 2 ms
    using namespace std::chrono_literals;
    timer_ = this->create_wall_timer(2ms, std::bind(&PotentialField2D::update, this)); //calls update() every 2ms

    RCLCPP_INFO(this->get_logger(), "potential_field_2d started (500 Hz).");
  }

private:
  void pose_callback(const geometry_msgs::msg::PoseArray::SharedPtr msg)
  {
    if (msg->poses.size() <= 5) { //ensure array has at least 6 element
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "PoseArray size=%zu; expected >= 6 (need poses[5]).", msg->poses.size());
      return;
    }
     // get robot position from 6th element
    cur_x_ = msg->poses[5].position.x;
    cur_y_ = msg->poses[5].position.y;
    got_pose_ = true; //flag if have valid pose data

    RCLCPP_INFO_ONCE(this->get_logger(), "Received first pose: (%.3f, %.3f)", cur_x_, cur_y_);
  }

  void service_callback(
    const std::shared_ptr<highlevel_interfaces::srv::Move2d::Request> request,
    std::shared_ptr<highlevel_interfaces::srv::Move2d::Response> response)
  {
    //false if service request is received before first pose feedback
    if (!got_pose_) {
      response->success = false;
      RCLCPP_WARN(this->get_logger(), "Service called before first pose; success=false.");
      return;
    }
    //set new target from service request
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

    //[vx, vy]
    vel_msg.data.resize(2);

    //if not ready publish (0,0) and done=true
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
    // vector to target
    const double dx = target_x_ - cur_x_;
    const double dy = target_y_ - cur_y_;
    const double dist = std::sqrt(dx * dx + dy * dy);

    //check if close enough to target
    done_msg.data = (dist < 0.1);

    //potential field velocity (attractive)
    double vx = katt * dx;
    double vy = katt * dy;

    //normalize only if norm > vmax
    const double vnorm = std::sqrt(vx * vx + vy * vy);
    if (vnorm > vmax && vnorm > 1e-9) {
      vx = (vx / vnorm) * vmax;
      vy = (vy / vnorm) * vmax;
    }

    //publish velocity and done status
    vel_msg.data[0] = vx;
    vel_msg.data[1] = vy;

    vel_pub_->publish(vel_msg);
    done_pub_->publish(done_msg);
  }

  //ROS interfaces
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr pose_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
  rclcpp::Service<highlevel_interfaces::srv::Move2d>::SharedPtr move_srv_;
  rclcpp::TimerBase::SharedPtr timer_;

  //flag
  bool got_pose_{false};
  bool got_target_{false};
  //pc
  double cur_x_{0.0};
  double cur_y_{0.0};
  //pt
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
