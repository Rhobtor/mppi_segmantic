from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    parameters = PathJoinSubstitution([
        get_package_share_directory("terrain_mppi_segmentation"),
        "config", "terrain_mppi_segmentation.yaml",
    ])
    semantic = LaunchConfiguration("semantic_enable")
    clock = {"use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool)}
    return LaunchDescription([
        DeclareLaunchArgument("semantic_enable", default_value="false"),
        DeclareLaunchArgument("use_sim_time", default_value="false"),
        DeclareLaunchArgument("mask_topic", default_value="/segmentation_cropped/mask"),
        DeclareLaunchArgument("cloud_topic", default_value="/camera/points"),
        DeclareLaunchArgument("goal_topic", default_value="/goal_ps"),
        DeclareLaunchArgument("goal_topic_pa", default_value="/goal_local"),
        DeclareLaunchArgument("cmd_topic", default_value="/cmd_vel"),
        DeclareLaunchArgument("fixed_frame", default_value="map"),
        Node(
            package="terrain_mppi_segmentation", executable="semantic_cost_node",
            name="semantic_cost_node", output="screen", condition=IfCondition(semantic),
            parameters=[parameters, clock, {
                "mask_topic": LaunchConfiguration("mask_topic"),
                "cloud_topic": LaunchConfiguration("cloud_topic"),
                "fixed_frame": LaunchConfiguration("fixed_frame"),
            }],
        ),
        Node(
            package="terrain_mppi_segmentation", executable="terrain_mppi_segmentation",
            name="terrain_mppi_segmentation", output="screen",
            parameters=[parameters, clock, {
                "semantic_enable": ParameterValue(semantic, value_type=bool),
                "goal_topic_ps": LaunchConfiguration("goal_topic"),
                "goal_topic_pa": LaunchConfiguration("goal_topic_pa"),
                "cmd_topic": LaunchConfiguration("cmd_topic"),
                "fixed_frame": LaunchConfiguration("fixed_frame"),
            }],
        ),
    ])
