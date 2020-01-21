#!/usr/bin/env python

"""
Node that repeatedly publishes the demo binvox.
This is useful for testing the visualization pipeline

"""

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import rospy
from shape_utils import vox_to_msg

import numpy as np
import binvox_rw
import rospkg

base_path = ''



def demo():
    '''
    Publish sample data to ROS
    '''
    global base_path

    rospy.init_node('shape_demo_loader')

    base_path = rospkg.RosPack().get_path('mps_shape_completion')

    # Read demo binvox as (64*64*64) array
    with open(base_path + '/demo/occupy.binvox', 'rb') as f:
        occ = binvox_rw.read_as_3d_array(f).data

    #Currently unoccupied is unused in this file
    with open(base_path + '/demo/non_occupy.binvox', 'rb') as f:
        non = binvox_rw.read_as_3d_array(f).data

        
    pub = rospy.Publisher('demo_voxel_grid', numpy_msg(Float32MultiArray), queue_size=10)
    while not rospy.is_shutdown():
        pub.publish(vox_to_msg(occ))
        rospy.sleep(1.0)


if __name__ == '__main__':
    demo()
