from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    kortex_description_share = get_package_share_directory("kortex_description")
    potential_field_share = get_package_share_directory("potential_field")

    return LaunchDescription([
        Node(
            package="dynamic_controller",
            executable="dynamic_controller",
            name="dynamic_controller",
            output="screen",
            parameters=[{
                "use_sim_time": True,
                "urdf_file_name": os.path.join(kortex_description_share, "robots", "gen3.urdf"),
                "publish_rate": 500.0,
                "with_redundancy": True,
                "linear_k": 25.0,
                "linear_d": 6.0,
                "joint_k": 100.0,
                "joint_d": 20.0,
            }],
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(potential_field_share, "launch", "potential_field_task.launch.xml")
            )
        ),
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(
                os.path.join(potential_field_share, "launch", "potential_field_joint.launch.xml")
            )
        ),
    ])
