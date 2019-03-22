#/bin/bash

. /opt/ros/kinetic/setup.bash
. /root/catkin_ws/devel/setup.bash
catkin_make run_tests && catkin_test_results
