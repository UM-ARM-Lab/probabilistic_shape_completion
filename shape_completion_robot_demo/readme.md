# Running on Victor

To run this on Victor (The armlab robot) you will need:

1. Start victor's control interface
- `roslauch roslaunch victor_description visualize_gripper.launch gui:=True joint_state_publisher:=True`
- `ssh realtime`, `roslaunch victor_hardware_interface dual_arm_lcm_bridge.launch`
- `ssh loki`, `roslaunch mps_launch_files kinect_vicon_real_robot.launch pov:="victor_head"`
2. `rviz` open `kinect_shape_completion.rviz`
3. Start the object segmentation algorithm (I used Armada)
- `cd catkin_ws/src/object_segmentation/scripts`, `./republish_segmentedinect_img.py`
4. Run the `demox` scripts.
