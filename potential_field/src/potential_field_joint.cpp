#include <array>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/trigger.hpp"

using std::placeholders::_1;
using std::placeholders::_2;

class PotentialFieldJoint : public rclcpp::Node
{
public:
  PotentialFieldJoint()
  : Node("potential_field_joint")
  {
    // Parameters
    this->declare_parameter<double>("publish_rate", 500.0);
    this->declare_parameter<double>("k_att", 5.0);
    this->declare_parameter<std::vector<double>>(
      "maximum_joint_velocity", std::vector<double>{1,1,1,1,1,1,1});
    this->declare_parameter<std::vector<double>>(
      "default_joint_position", std::vector<double>{0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0});
    this->declare_parameter<double>("done_threshold", 0.05);

    const double publish_rate = this->get_parameter("publish_rate").as_double();
    k_att_ = this->get_parameter("k_att").as_double();
    done_threshold_ = this->get_parameter("done_threshold").as_double();

    // Read vectors and validate sizes
    std::vector<double> vmax_vec = this->get_parameter("maximum_joint_velocity").as_double_array();
    std::vector<double> qdef_vec = this->get_parameter("default_joint_position").as_double_array();

    if (vmax_vec.size() != DOF) {
      RCLCPP_WARN(this->get_logger(),
        "maximum_joint_velocity size=%zu (expected 7). Using default.", vmax_vec.size());
      vmax_vec = {1,1,1,1,1,1,1};
    }
    if (qdef_vec.size() != DOF) {
      RCLCPP_WARN(this->get_logger(),
        "default_joint_position size=%zu (expected 7). Using default.", qdef_vec.size());
      qdef_vec = {0.0, 0.6, 0.0, 0.6, 0.0, 0.6, 0.0};
    }

    for (size_t i = 0; i < DOF; ++i) {
      vmax_[i] = vmax_vec[i];
      q_default_[i] = qdef_vec[i];
    }

    RCLCPP_INFO(this->get_logger(),
      "params: rate=%.1f Hz k_att=%.3f done_th=%.3f",
      publish_rate, k_att_, done_threshold_);

    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

    // Sub
    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", qos, std::bind(&PotentialFieldJoint::joint_callback, this, _1));

    // Pubs (A3 + A4)
    joint_vel_pub_a3_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/planner/joint_velocity", qos);
    joint_pos_pub_a5_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/gen3/reference/position", qos);
    joint_vel_pub_a4_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/gen3/reference/velocity", qos);

    done_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/planner/done", qos);

    // (A3 compatibility) Homing service:
    homing_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/planner/homing", std::bind(&PotentialFieldJoint::homing_callback, this, _1, _2));

    joint_vel_msg_.data.resize(DOF);
    joint_pos_msg_.data.resize(DOF);
    for (size_t i = 0; i < DOF; ++i) {
      joint_pos_msg_.data[i] = q_default_[i];
    }

    // Timer
    auto timer_period = std::chrono::duration<double>(1.0 / publish_rate);
    timer_ = this->create_wall_timer(timer_period, std::bind(&PotentialFieldJoint::update, this));

    // Before the first joint feedback, keep publishing the desired joint target so the
    // redundancy controller always has a valid q_ref.
    publish_planner_zero_and_done_true();

    RCLCPP_INFO(this->get_logger(), "potential_field_joint started (%.1f Hz).", publish_rate);
  }

private:
  static constexpr size_t DOF = 7;

  void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    if (msg->position.size() < DOF) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "JointState position size=%zu < 7. Waiting...", msg->position.size());
      have_joint_ = false;
      return;
    }

    //assume first 7 positions correspond to joints
    for (size_t i = 0; i < DOF; ++i) q_[i] = msg->position[i];

    have_joint_ = true;
    RCLCPP_INFO_ONCE(this->get_logger(), "Received first /joint_states.");

    // If want A3 “homing” semantics,can auto-enable it on first feedback:
    if (!homing_ever_triggered_) {
      homing_active_ = true;
    }
  }

  void homing_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    if (!have_joint_) {
      response->success = false;
      response->message = "No /joint_states yet.";
      return;
    }
    homing_active_ = true;          // A3 “start homing”
    homing_ever_triggered_ = true;
    response->success = true;
    response->message = "Homing enabled.";
    RCLCPP_INFO(this->get_logger(), "Homing enabled via /planner/homing.");
  }

  void update()
  {
    // Keep /gen3/reference/position available at all times for A5.
    if (!have_joint_) {
      publish_planner_zero_and_done_true();
      return;
    }

    // Compute qdot toward default (potential field)
    bool all_within = true;

    for (size_t i = 0; i < DOF; ++i) {
      const double e = q_default_[i] - q_[i];
      if (std::fabs(e) >= done_threshold_) all_within = false;

      double qdot = k_att_ * e;

      const double lim = std::fabs(vmax_[i]);
      qdot = clamp(qdot, -lim, lim);

      // If want A3(only move when homing_active_), apply here:
      // For A4 can just always move; keeping for future reuse.
      if (!homing_active_) qdot = 0.0;

      joint_vel_msg_.data[i] = qdot;
    }

    done_msg_.data = all_within;

    // If "done", optionally stop homing (A3
    if (all_within) {
      homing_active_ = false;
      for (size_t i = 0; i < DOF; ++i) joint_vel_msg_.data[i] = 0.0;
    }

    // q_ref is the fixed null-space target from the assignment.
    for (size_t i = 0; i < DOF; ++i) {
      joint_pos_msg_.data[i] = q_default_[i];
    }

    joint_vel_pub_a3_->publish(joint_vel_msg_);
    joint_pos_pub_a5_->publish(joint_pos_msg_);
    done_pub_->publish(done_msg_);
    joint_vel_pub_a4_->publish(joint_vel_msg_);
  }

  void publish_planner_zero_and_done_true()
  {
    for (size_t i = 0; i < DOF; ++i) joint_vel_msg_.data[i] = 0.0;
    for (size_t i = 0; i < DOF; ++i) joint_pos_msg_.data[i] = q_default_[i];
    done_msg_.data = true;

    joint_vel_pub_a3_->publish(joint_vel_msg_);
    joint_pos_pub_a5_->publish(joint_pos_msg_);
    done_pub_->publish(done_msg_);
    //no publish to /gen3/reference/velocity ici
  }

  static double clamp(double v, double lo, double hi)
  {
    return std::max(lo, std::min(v, hi));
  }

  //ros interfaces
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_pub_a3_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_pos_pub_a5_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_pub_a4_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr homing_srv_;
  rclcpp::TimerBase::SharedPtr timer_;

  //msg
  std_msgs::msg::Float64MultiArray joint_vel_msg_;
  std_msgs::msg::Float64MultiArray joint_pos_msg_;
  std_msgs::msg::Bool done_msg_;

  //state
  bool have_joint_{false};

  //cumulative compatibility with a3:
  bool homing_active_{false};
  bool homing_ever_triggered_{false};

  std::array<double, DOF> q_{};
  std::array<double, DOF> q_default_{};
  std::array<double, DOF> vmax_{};

  double k_att_{5.0};
  double done_threshold_{0.05};
};

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PotentialFieldJoint>());
  rclcpp::shutdown();
  return 0;
}
