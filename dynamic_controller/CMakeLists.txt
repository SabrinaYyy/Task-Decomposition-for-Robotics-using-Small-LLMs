cmake_minimum_required(VERSION 3.8)
project(dynamic_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(highlevel_interfaces REQUIRED)
find_package(pinocchio REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(dynamic_controller src/dynamic_controller.cpp)

ament_target_dependencies(dynamic_controller
  geometry_msgs
  highlevel_interfaces
  rclcpp
  sensor_msgs
  std_msgs
)

target_link_libraries(dynamic_controller
  Eigen3::Eigen
  pinocchio::pinocchio
)

install(TARGETS
  dynamic_controller
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()
