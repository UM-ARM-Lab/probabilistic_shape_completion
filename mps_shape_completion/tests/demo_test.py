#!/usr/bin/env python

import unittest
import rostest
import rospy
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float32MultiArray
import os
import rospkg

class RosDemoTest(unittest.TestCase):
    def test_ros_demo(self):
        base_path = rospkg.RosPack().get_path('mps_shape_completion')
        path = base_path + '/demo/output.binvox'
        self.assertFalse(os.path.isfile(path))

        while(not os.path.isfile(base_path + '/demo/output.binvox')):
            rospy.sleep(1)



if __name__ == '__main__':
    rostest.rosrun('mps_shape_completion', 'RosDemo', RosDemoTest)
