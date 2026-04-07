from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    control_orientation = LaunchConfiguration("control_orientation")

    kinematic_controller_share = get_package_share_directory("kinematic_controller")

    return LaunchDescription([
        DeclareLaunchArgument(
            "control_orientation",
            default_value="true",
            description="Use 6D pose control when true, or 3D position-only control when false.",
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(kinematic_controller_share, "launch", "3d.launch.xml")
            ),
            condition=UnlessCondition(control_orientation),
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(kinematic_controller_share, "launch", "6d.launch.xml")
            ),
            condition=IfCondition(control_orientation),
        ),
    ])
