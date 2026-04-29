import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # I keep these as launch arguments so I can test the same launch file in
    # position-only mode or full 6D mode, even though the final demo uses 3D.
    use_sim_time = LaunchConfiguration("use_sim_time")
    control_orientation = LaunchConfiguration("control_orientation")
    run_executor = LaunchConfiguration("run_executor")
    executor_script = LaunchConfiguration("executor_script")
    task_json = LaunchConfiguration("task_json")

    llm_project_share = get_package_share_directory("llm_project")
    kortex_description_share = get_package_share_directory("kortex_description")
    kinematic_controller_share = get_package_share_directory("kinematic_controller")

    default_task_json = os.path.join(llm_project_share, "llm", "task.json")
    default_executor_script = os.path.join(llm_project_share, "llm", "ros_executor.py")

    return LaunchDescription([
        # Gazebo needs these paths to find the Kortex pick-and-place world and
        # any local llm_project models/worlds without relying on my shell setup.
        AppendEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            os.path.join(kortex_description_share, "models"),
        ),
        AppendEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            os.path.join(kortex_description_share, "worlds"),
        ),
        AppendEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            os.path.join(llm_project_share, "models"),
        ),
        AppendEnvironmentVariable(
            "GZ_SIM_RESOURCE_PATH",
            os.path.join(llm_project_share, "worlds"),
        ),

        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use the Gazebo simulation clock.",
        ),
        DeclareLaunchArgument(
            "control_orientation",
            default_value="false",
            description="Use 6D end-effector pose control when true, or 3D position control when false.",
        ),
        DeclareLaunchArgument(
            "run_executor",
            default_value="false",
            description="Run the JSON-to-ROS executor script after launching simulation/control.",
        ),
        DeclareLaunchArgument(
            "executor_script",
            default_value=default_executor_script,
            description="Path to the Python executor that reads task_json and calls ROS 2 actions.",
        ),
        DeclareLaunchArgument(
            "task_json",
            default_value=default_task_json,
            description="Path to the LLM-generated JSON plan.",
        ),

        # I launch the provided Kortex simulation first. This gives me the robot,
        # table, cup, cube, ros2_control, and the gripper controller interface.
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare("kortex_sim"),
                    "launch",
                    "final.launch.py",
                ])
            ),
            launch_arguments={"use_sim_time": use_sim_time}.items(),
        ),

        # I explicitly spawn the gripper controller so the executor can call
        # /gripper_controller/gripper_cmd.
        Node(
            package="controller_manager",
            executable="spawner",
            arguments=["gripper_controller", "-c", "/controller_manager"],
        ),

        # The final behavior worked reliably with 3D position control and no
        # redundancy. I keep the 6D branch available for debugging.
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(kinematic_controller_share, "launch", "6d.launch.xml")
            ),
            condition=IfCondition(control_orientation),
        ),

        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(kinematic_controller_share, "launch", "3d.launch.xml")
            ),
            condition=UnlessCondition(control_orientation),
        ),

        ExecuteProcess(
            # This is optional because I usually launch the simulator first and
            # run the text-to-robot script from a second terminal.
            cmd=["python3", executor_script, "--task-json", task_json],
            output="screen",
            condition=IfCondition(run_executor),
        ),
    ])
