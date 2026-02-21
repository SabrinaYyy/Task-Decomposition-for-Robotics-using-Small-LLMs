cmake_minimum_required(VERSION 3.8)
project(kinematic_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pinocchio REQUIRED)
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# Add executable
add_executable(kinematic_controller
  src/kinematic_controller.cpp
)
# Link dependencies
ament_target_dependencies(kinematic_controller
  rclcpp
  sensor_msgs
  geometry_msgs
  std_msgs
)

target_link_libraries(kinematic_controller
  pinocchio::pinocchio
  Eigen3::Eigen
)

# Install targets
install(TARGETS
  kinematic_controller
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include
)

ament_package()