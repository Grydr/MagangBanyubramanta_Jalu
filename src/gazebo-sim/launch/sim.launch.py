import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration, Command
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():

	use_sim_time = LaunchConfiguration('use_sim_time', default=True)

	pkg_name = 'sim' # change this

	robot_state_publisher = Node(
		package='robot_state_publisher',
		executable='robot_state_publisher',
		parameters=[{
			'robot_description': Command([
				'xacro ', os.path.join(get_package_share_directory(pkg_name), 'description', 'robot.xacro'), 
				' sim_mode:=', use_sim_time
			])
		}]
	)

	gazebo = IncludeLaunchDescription(
		PythonLaunchDescriptionSource([os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')]),
		launch_arguments={'world': os.path.join(get_package_share_directory(pkg_name), 'worlds', 'zero_g.world')}.items() # change / remove this
	)

	spawn_entity = Node(
		package='gazebo_ros',
		executable='spawn_entity.py',
		arguments=[
			'-topic', 'robot_description',
			'-entity', 'ronald' # change this (optional)
		],
		output='screen'
	)

	return LaunchDescription([
		DeclareLaunchArgument(
			'use_sim_time',
			default_value='true',
			description='Use simulation time'
		),
		robot_state_publisher,
		gazebo,
		spawn_entity,
	])