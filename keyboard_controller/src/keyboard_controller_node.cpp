#include <string>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"

using std::placeholders::_1;

class KeyboardControllerNode : public rclcpp::Node
{
public:
  KeyboardControllerNode()
  : Node("keyboard_controller_node")
  {
    key_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/keyboard_reader/cmd", 10,
      std::bind(&KeyboardControllerNode::on_key, this, _1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/husky/cmd_vel", 10);

    // 100Hz
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&KeyboardControllerNode::publish_cmd, this));

    RCLCPP_INFO(this->get_logger(),
      "keyboard_controller_node running @100Hz. Sub: /keyboard_reader/cmd  Pub: /husky/cmd_vel");
  }

private:
  void on_key(const std_msgs::msg::String::SharedPtr msg)
  {
    last_key_ = msg->data;
  }

  void publish_cmd()
  {
    geometry_msgs::msg::TwistStamped out;
    out.header.stamp = this->get_clock()->now();
    out.header.frame_id = "base_link";

    //zero vector
    out.twist.linear.x = 0.0;
    out.twist.linear.y = 0.0;
    out.twist.linear.z = 0.0;
    out.twist.angular.x = 0.0;
    out.twist.angular.y = 0.0;
    out.twist.angular.z = 0.0;

    if (!last_key_.empty()) {
      const char k = last_key_[0];
      if (k == 'i') {
        out.twist.linear.x = 0.5;
      } else if (k == 'u') {
        out.twist.linear.x = 0.5;
        out.twist.angular.z = 0.5;
      } else if (k == 'o') {
        out.twist.linear.x = 0.5;
        out.twist.angular.z = -0.5;
      }
    }

    cmd_pub_->publish(out);
  }

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr key_sub_;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  std::string last_key_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<KeyboardControllerNode>());
  rclcpp::shutdown();
  return 0;
}
