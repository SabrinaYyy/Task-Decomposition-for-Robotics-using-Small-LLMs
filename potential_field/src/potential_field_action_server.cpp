#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <rclcpp/create_timer.hpp>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <highlevel_interfaces/action/pose_command.hpp>

#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <thread>

class PotentialFieldActionServer : public rclcpp::Node
{
public:
  using PoseCommand = highlevel_interfaces::action::PoseCommand;
  using GoalHandlePoseCommand = rclcpp_action::ServerGoalHandle<PoseCommand>;

  PotentialFieldActionServer()
  : Node("potential_field_action_server"),
    first_feedback_received_(false),
    has_active_goal_(false),
    current_position_(Eigen::Vector3d::Zero()),
    target_position_(Eigen::Vector3d(0.5, 0.2, 0.5)),
    current_orientation_(Eigen::Quaterniond::Identity()),
    target_orientation_(rpyToQuaternion(0.0, 0.0, 0.0))
  {
    this->declare_parameter<double>("publish_rate", 500.0);
    this->declare_parameter<double>("k_att_linear", 5.0);
    this->declare_parameter<double>("k_att_angular", 5.0);
    this->declare_parameter<double>("maximum_linear_velocity", 1.0);
    this->declare_parameter<double>("maximum_angular_velocity", 1.5);

    this->get_parameter("publish_rate", publish_rate_);
    this->get_parameter("k_att_linear", k_att_linear_);
    this->get_parameter("k_att_angular", k_att_angular_);
    this->get_parameter("maximum_linear_velocity", max_linear_velocity_);
    this->get_parameter("maximum_angular_velocity", max_angular_velocity_);

    pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>(
      "/gen3/feedback/pose",
      10,
      std::bind(&PotentialFieldActionServer::poseCallback, this, std::placeholders::_1));

    twist_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(
      "/gen3/reference/twist",
      10);

    timer_ = rclcpp::create_timer(
      this->get_node_base_interface(),
      this->get_node_timers_interface(),
      this->get_clock(),
      std::chrono::duration<double>(1.0 / publish_rate_),
      std::bind(&PotentialFieldActionServer::timerCallback, this));

    action_server_ = rclcpp_action::create_server<PoseCommand>(
      this,
      "/planner/move_pose",
      std::bind(&PotentialFieldActionServer::handleGoal, this, std::placeholders::_1, std::placeholders::_2),
      std::bind(&PotentialFieldActionServer::handleCancel, this, std::placeholders::_1),
      std::bind(&PotentialFieldActionServer::handleAccepted, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(), "Potential field action server initialized");
  }

private:
  static Eigen::Quaterniond rpyToQuaternion(double roll, double pitch, double yaw)
  {
    const Eigen::AngleAxisd roll_rotation(roll, Eigen::Vector3d::UnitX());
    const Eigen::AngleAxisd pitch_rotation(pitch, Eigen::Vector3d::UnitY());
    const Eigen::AngleAxisd yaw_rotation(yaw, Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond quaternion = yaw_rotation * pitch_rotation * roll_rotation;
    quaternion.normalize();
    return quaternion;
  }

  static Eigen::Quaterniond poseOrientationToEigen(const geometry_msgs::msg::Pose & pose)
  {
    Eigen::Quaterniond quaternion(
      pose.orientation.w,
      pose.orientation.x,
      pose.orientation.y,
      pose.orientation.z);

    if (quaternion.norm() < 1e-9) {
      return Eigen::Quaterniond::Identity();
    }

    quaternion.normalize();
    return quaternion;
  }

  static Eigen::Vector3d saturateVector(const Eigen::Vector3d & vector, double max_norm)
  {
    const double norm = vector.norm();
    if (norm < 1e-9 || norm <= max_norm) {
      return vector;
    }
    return vector * (max_norm / norm);
  }

  bool isGoalWithinBounds(const PoseCommand::Goal & goal) const
  {
    const bool xy_ok =
      (goal.x >= -0.8 && goal.x <= 0.8) &&
      (goal.y >= -0.8 && goal.y <= 0.8);
    const bool z_ok = (goal.z >= 0.01 && goal.z <= 0.8);
    const bool orientation_ok =
      (goal.roll >= -3.14 && goal.roll <= 3.14) &&
      (goal.pitch >= -3.14 && goal.pitch <= 3.14) &&
      (goal.yaw >= -3.14 && goal.yaw <= 3.14);

    return xy_ok && z_ok && orientation_ok;
  }

  geometry_msgs::msg::Twist computeReferenceTwist(
    const Eigen::Vector3d & current_position,
    const Eigen::Quaterniond & current_orientation,
    const Eigen::Vector3d & target_position,
    const Eigen::Quaterniond & target_orientation,
    double & distance_translation,
    double & distance_orientation) const
  {
    const Eigen::Vector3d position_error = target_position - current_position;
    Eigen::Vector3d linear_velocity = k_att_linear_ * position_error;
    linear_velocity = saturateVector(linear_velocity, max_linear_velocity_);
    distance_translation = position_error.norm();

    Eigen::Quaterniond quaternion_error = target_orientation * current_orientation.conjugate();
    quaternion_error.normalize();

    const double angle = current_orientation.angularDistance(target_orientation);
    distance_orientation = angle;

    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    if (angle > 1e-6) {
      const Eigen::AngleAxisd angle_axis(quaternion_error);
      const double sign = (target_orientation.dot(current_orientation) < 0.0) ? -1.0 : 1.0;
      angular_velocity = sign * k_att_angular_ * angle_axis.angle() * angle_axis.axis();
    }

    geometry_msgs::msg::Twist twist_msg;
    twist_msg.linear.x = linear_velocity.x();
    twist_msg.linear.y = linear_velocity.y();
    twist_msg.linear.z = linear_velocity.z();
    twist_msg.angular.x = angular_velocity.x();
    twist_msg.angular.y = angular_velocity.y();
    twist_msg.angular.z = angular_velocity.z();
    return twist_msg;
  }

  void setTargetPose(
    const Eigen::Vector3d & target_position,
    const Eigen::Quaterniond & target_orientation)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    target_position_ = target_position;
    target_orientation_ = target_orientation;
  }

  void setDefaultTarget()
  {
    setTargetPose(Eigen::Vector3d(0.5, 0.2, 0.5), rpyToQuaternion(0.0, 0.0, 0.0));
  }

  rclcpp_action::GoalResponse handleGoal(
    const rclcpp_action::GoalUUID &,
    std::shared_ptr<const PoseCommand::Goal> goal)
  {
    if (!isGoalWithinBounds(*goal)) {
      RCLCPP_WARN(this->get_logger(), "Rejected action goal: target pose out of bounds");
      return rclcpp_action::GoalResponse::REJECT;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (has_active_goal_) {
      RCLCPP_WARN(this->get_logger(), "Rejected action goal: another goal is still active");
      return rclcpp_action::GoalResponse::REJECT;
    }

    RCLCPP_INFO(
      this->get_logger(),
      "Accepted goal: position=[%.3f %.3f %.3f], rpy=[%.3f %.3f %.3f]",
      goal->x, goal->y, goal->z, goal->roll, goal->pitch, goal->yaw);
    return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
  }

  rclcpp_action::CancelResponse handleCancel(
    const std::shared_ptr<GoalHandlePoseCommand> goal_handle)
  {
    (void)goal_handle;
    RCLCPP_INFO(this->get_logger(), "Received request to cancel active goal");
    return rclcpp_action::CancelResponse::ACCEPT;
  }

  void handleAccepted(const std::shared_ptr<GoalHandlePoseCommand> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      has_active_goal_ = true;
      active_goal_start_time_ = this->get_clock()->now();
      target_position_ = Eigen::Vector3d(goal->x, goal->y, goal->z);
      target_orientation_ = rpyToQuaternion(goal->roll, goal->pitch, goal->yaw);
    }

    std::thread(
      std::bind(&PotentialFieldActionServer::executeAction, this, std::placeholders::_1),
      goal_handle).detach();
  }

  void executeAction(const std::shared_ptr<GoalHandlePoseCommand> goal_handle)
  {
    const auto goal = goal_handle->get_goal();
    const Eigen::Vector3d target_position(goal->x, goal->y, goal->z);
    const Eigen::Quaterniond target_orientation = rpyToQuaternion(goal->roll, goal->pitch, goal->yaw);

    auto feedback = std::make_shared<PoseCommand::Feedback>();
    auto result = std::make_shared<PoseCommand::Result>();

    rclcpp::Rate rate(50.0);
    while (rclcpp::ok()) {
      if (goal_handle->is_canceling()) {
        result->success = false;
        result->message = "Goal canceled";
        goal_handle->canceled(result);
        setDefaultTarget();

        std::lock_guard<std::mutex> lock(mutex_);
        has_active_goal_ = false;
        return;
      }

      Eigen::Vector3d current_position;
      Eigen::Quaterniond current_orientation;
      rclcpp::Time start_time(0, 0, this->get_clock()->get_clock_type());
      bool feedback_ready = false;

      {
        std::lock_guard<std::mutex> lock(mutex_);
        current_position = current_position_;
        current_orientation = current_orientation_;
        start_time = active_goal_start_time_;
        feedback_ready = first_feedback_received_;
      }

      if (!feedback_ready) {
        rate.sleep();
        continue;
      }

      double distance_translation = 0.0;
      double distance_orientation = 0.0;
      const auto twist_msg = computeReferenceTwist(
        current_position,
        current_orientation,
        target_position,
        target_orientation,
        distance_translation,
        distance_orientation);

      (void)twist_msg;

      const double time_elapsed = (this->get_clock()->now() - start_time).seconds();

      feedback->distance_translation = distance_translation;
      feedback->distance_orientation = distance_orientation;
      feedback->time_elapsed = time_elapsed;
      goal_handle->publish_feedback(feedback);

      if (distance_translation < 0.05 && distance_orientation < 0.1) {
        result->success = true;
        result->message = "Goal reached";
        goal_handle->succeed(result);

        std::lock_guard<std::mutex> lock(mutex_);
        has_active_goal_ = false;
        return;
      }

      if (time_elapsed > 5.0) {
        result->success = false;
        result->message = "Goal aborted after timeout";
        goal_handle->abort(result);
        setDefaultTarget();

        std::lock_guard<std::mutex> lock(mutex_);
        has_active_goal_ = false;
        return;
      }

      rate.sleep();
    }

    result->success = false;
    result->message = "Node shutting down";
    if (rclcpp::ok()) {
      goal_handle->abort(result);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    has_active_goal_ = false;
  }

  void poseCallback(const geometry_msgs::msg::Pose::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    current_position_.x() = msg->position.x;
    current_position_.y() = msg->position.y;
    current_position_.z() = msg->position.z;
    current_orientation_ = poseOrientationToEigen(*msg);

    if (!first_feedback_received_) {
      first_feedback_received_ = true;
      RCLCPP_INFO(this->get_logger(), "First pose feedback received");
    }
  }

  void timerCallback()
  {
    Eigen::Vector3d current_position;
    Eigen::Vector3d target_position;
    Eigen::Quaterniond current_orientation;
    Eigen::Quaterniond target_orientation;
    bool feedback_ready = false;

    {
      std::lock_guard<std::mutex> lock(mutex_);
      current_position = current_position_;
      current_orientation = current_orientation_;
      target_position = target_position_;
      target_orientation = target_orientation_;
      feedback_ready = first_feedback_received_;
    }

    if (!feedback_ready) {
      return;
    }

    double distance_translation = 0.0;
    double distance_orientation = 0.0;
    const auto twist_msg = computeReferenceTwist(
      current_position,
      current_orientation,
      target_position,
      target_orientation,
      distance_translation,
      distance_orientation);

    twist_pub_->publish(twist_msg);
  }

  std::mutex mutex_;

  rclcpp::Subscription<geometry_msgs::msg::Pose>::SharedPtr pose_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr twist_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp_action::Server<PoseCommand>::SharedPtr action_server_;

  bool first_feedback_received_;
  bool has_active_goal_;

  Eigen::Vector3d current_position_;
  Eigen::Vector3d target_position_;
  Eigen::Quaterniond current_orientation_;
  Eigen::Quaterniond target_orientation_;

  rclcpp::Time active_goal_start_time_{0, 0, RCL_ROS_TIME};

  double publish_rate_{500.0};
  double k_att_linear_{5.0};
  double k_att_angular_{2.0};
  double max_linear_velocity_{1.0};
  double max_angular_velocity_{1.5};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<PotentialFieldActionServer>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
