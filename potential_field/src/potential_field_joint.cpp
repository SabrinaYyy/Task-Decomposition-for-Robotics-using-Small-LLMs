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
    // =========================
    // Parameters (A3)
    // =========================
    // Declare with DEFAULTS required by assignment
    // (If YAML provides them, they override automatically.) :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
    this->declare_parameter<double>("k_att", 5.0);
    this->declare_parameter<std::vector<double>>("maximum_joint_velocity",
      std::vector<double>{1, 1, 1, 1, 1, 1, 1});
    this->declare_parameter<std::vector<double>>("default_joint_position",
      std::vector<double>{0.0, 1.54, 0.0, 1.54, 0.0, 1.54, 0.0});
    this->declare_parameter<double>("done_threshold", 0.05);

    // Read parameters once (recommended: parameters during init) :contentReference[oaicite:5]{index=5}
    k_att_ = this->get_parameter("k_att").as_double();

    auto vmax_vec = this->get_parameter("maximum_joint_velocity").as_double_array();
    auto qdef_vec = this->get_parameter("default_joint_position").as_double_array();

    done_threshold_ = this->get_parameter("done_threshold").as_double();

    // Validate sizes (must be 7). If wrong, fall back to defaults.
    if (vmax_vec.size() != DOF) {
      RCLCPP_WARN(this->get_logger(),
                  "Param maximum_joint_velocity size=%zu (expected 7). Using default [1..].",
                  vmax_vec.size());
      vmax_ = {1,1,1,1,1,1,1};
    } else {
      for (size_t i = 0; i < DOF; ++i) vmax_[i] = vmax_vec[i];
    }

    if (qdef_vec.size() != DOF) {
      RCLCPP_WARN(this->get_logger(),
                  "Param default_joint_position size=%zu (expected 7). Using default.",
                  qdef_vec.size());
      q_default_ = {0.0, 1.54, 0.0, 1.54, 0.0, 1.54, 0.0};
    } else {
      for (size_t i = 0; i < DOF; ++i) q_default_[i] = qdef_vec[i];
    }

    RCLCPP_INFO(this->get_logger(),
                "potential_field_joint params loaded: k_att=%.3f done_th=%.3f",
                k_att_, done_threshold_);

    // =========================
    // Pub/Sub/Service
    // =========================
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));

    joint_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", qos,
      std::bind(&PotentialFieldJoint::joint_callback, this, _1));

    joint_vel_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
      "/planner/joint_velocity", qos);

    done_pub_ = this->create_publisher<std_msgs::msg::Bool>(
      "/planner/done", qos);

    homing_srv_ = this->create_service<std_srvs::srv::Trigger>(
      "/planner/homing",
      std::bind(&PotentialFieldJoint::homing_callback, this, _1, _2));

    // Pre-size messages once (avoid resizing in every loop) :contentReference[oaicite:6]{index=6}
    joint_vel_msg_.data.resize(DOF);

    // 500 Hz loop (2 ms) as required :contentReference[oaicite:7]{index=7}
    using namespace std::chrono_literals;
    timer_ = this->create_wall_timer(2ms, std::bind(&PotentialFieldJoint::update, this));

    // Required initial behavior: output zero velocity & done=true before activation :contentReference[oaicite:8]{index=8}
    publish_zero_and_done_true();

    RCLCPP_INFO(this->get_logger(), "potential_field_joint started (500 Hz).");
  }

private:
  static constexpr size_t DOF = 7;

  // ---------- Callbacks ----------
  void joint_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
  {
    // We need at least 7 joint positions
    if (msg->position.size() < DOF) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "JointState position size=%zu < 7. Waiting...",
                           msg->position.size());
      have_joint_ = false;
      return;
    }

    // Assume first 7 positions correspond to the 7 arm joints (typical for course sim)
    for (size_t i = 0; i < DOF; ++i) {
      q_[i] = msg->position[i];
    }

    have_joint_ = true;
    RCLCPP_INFO_ONCE(this->get_logger(), "Received first /joint_states.");
  }

  void homing_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    // Requirement: service returns success=false if called before first joint feedback :contentReference[oaicite:9]{index=9}
    if (!have_joint_) {
      response->success = false;
      response->message = "No joint feedback yet (/joint_states not received).";
      RCLCPP_WARN(this->get_logger(), "/planner/homing called before first /joint_states => success=false");
      return;
    }

    homing_active_ = true;
    response->success = true;
    response->message = "Homing started.";
    RCLCPP_INFO(this->get_logger(), "Homing triggered.");
  }

  // ---------- Main Loop ----------
  void update()
  {
    // Requirement: before service call OR before joint feedback:
    // publish zero velocity and done=true :contentReference[oaicite:10]{index=10}
    if (!have_joint_ || !homing_active_) {
      publish_zero_and_done_true();
      return;
    }

    // Compute per-joint error and velocity command:
    // qdot = k_att * (q_default - q)
    // Then cap EACH joint by maximum_joint_velocity (no normalization) :contentReference[oaicite:11]{index=11}
    bool all_within = true;

    for (size_t i = 0; i < DOF; ++i) {
      const double e = q_default_[i] - q_[i];

      if (std::fabs(e) >= done_threshold_) {
        all_within = false;
      }

      double qdot = k_att_ * e;

      // per-joint clamp
      const double lim = std::fabs(vmax_[i]);
      qdot = clamp(qdot, -lim, lim);

      joint_vel_msg_.data[i] = qdot;
    }

    done_msg_.data = all_within;

    // If done, you can optionally stop homing (safe behavior)
    // The assignment mainly checks /planner/done and velocity. :contentReference[oaicite:12]{index=12}
    if (all_within) {
      homing_active_ = false;
      // publish zeros (so it doesn't drift)
      for (size_t i = 0; i < DOF; ++i) joint_vel_msg_.data[i] = 0.0;
    }

    joint_vel_pub_->publish(joint_vel_msg_);
    done_pub_->publish(done_msg_);
  }

  // ---------- Helpers ----------
  static double clamp(double v, double lo, double hi)
  {
    return std::max(lo, std::min(v, hi));
  }

  void publish_zero_and_done_true()
  {
    for (size_t i = 0; i < DOF; ++i) {
      joint_vel_msg_.data[i] = 0.0;
    }
    done_msg_.data = true;

    joint_vel_pub_->publish(joint_vel_msg_);
    done_pub_->publish(done_msg_);
  }

  // ---------- ROS interfaces ----------
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr joint_vel_pub_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr done_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr homing_srv_;
  rclcpp::TimerBase::SharedPtr timer_;

  // ---------- Messages (reused, avoid allocations in loop) ----------
  std_msgs::msg::Float64MultiArray joint_vel_msg_;
  std_msgs::msg::Bool done_msg_;

  // ---------- State ----------
  bool have_joint_{false};
  bool homing_active_{false};

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
