#include <string> //std::string last_key_
#include <chrono> //std::chrono::milliseconds(10) for the timer period

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp" //message type for incoming key commands
#include "geometry_msgs/msg/twist_stamped.hpp" //message type for outgoing velocity commands

using std::placeholders::_1; //for std::bind() to write _1 as the first callback argument placeholder

class KeyboardControllerNode : public rclcpp::Node //Ros2 node class
{
public:
  KeyboardControllerNode()
  : Node("keyboard_controller_node") //initialize the base rclcpp::Node with that node name
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10));
    key_sub_ = this->create_subscription<std_msgs::msg::String>(
      "/keyboard_reader/cmd", qos,
      std::bind(&KeyboardControllerNode::on_key, this, _1));//subscriber to /keyboard_reader/cmd

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
      "/husky/cmd_vel", rclcpp::QoS(1)); //publisher send TwistStamped to /husky/cmd_vel

    // 100Hz
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(10),
      std::bind(&KeyboardControllerNode::publish_cmd, this)); //wall timer that triggers every 10ms(100Hz)

    RCLCPP_INFO(this->get_logger(),
      "keyboard_controller_node running @100Hz.");
  }

private:
  void on_key(const std_msgs::msg::String::SharedPtr msg)
  {
    last_key_ = msg->data;
  }//When a key message arrive, store in last_key_

  void publish_cmd()
  {
    geometry_msgs::msg::TwistStamped out; //create a message out
    out.header.stamp = this->get_clock()->now(); //timestamp
    out.header.frame_id = "base_link";

    //zero vector
    out.twist.linear.x = 0.0;
    out.twist.linear.y = 0.0;
    out.twist.linear.z = 0.0;
    out.twist.angular.x = 0.0;
    out.twist.angular.y = 0.0;
    out.twist.angular.z = 0.0;

    if (!last_key_.empty()) {
      const char k = last_key_[0]; //take the first char as the command
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

    cmd_pub_->publish(out); //publish the command every timer tick
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
