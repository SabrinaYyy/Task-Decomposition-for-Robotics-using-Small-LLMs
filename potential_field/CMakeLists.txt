cmake_minimum_required(VERSION 3.8)
project(potential_field)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)   #JointState
find_package(std_srvs REQUIRED)      #Trigger service
find_package(highlevel_interfaces REQUIRED)
find_package(Eigen3 REQUIRED)        # ADD THIS - needed for potential_field_task

# a2
add_executable(potential_field_2d src/potential_field_2d.cpp)
ament_target_dependencies(potential_field_2d
  rclcpp
  geometry_msgs
  std_msgs
  highlevel_interfaces
)

# a3
add_executable(potential_field_joint src/potential_field_joint.cpp)
ament_target_dependencies(potential_field_joint
  rclcpp
  std_msgs
  sensor_msgs
  std_srvs
)
target_link_libraries(potential_field_joint
  Eigen3::Eigen
)

# a4
add_executable(potential_field_task src/potential_field_task.cpp)
ament_target_dependencies(potential_field_task
  rclcpp
  geometry_msgs
  std_msgs
  highlevel_interfaces
)

target_link_libraries(potential_field_task
  Eigen3::Eigen
)


install(
  DIRECTORY config launch    # ADD 'launch' here to install launch files
  DESTINATION share/${PROJECT_NAME}
)

install(TARGETS 
  potential_field_2d
  potential_field_joint
  potential_field_task
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()