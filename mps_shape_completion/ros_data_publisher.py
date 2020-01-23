#!/usr/bin/env python

"""
Cycles through and publishes data for visualization
"""

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import rospy
from shape_utils import vox_to_msg

import numpy as np
import binvox_rw
import rospkg
import os

base_path = ''



def demo():
    '''
    Publish sample data to ROS
    '''
    global base_path

    rospy.init_node('shape_demo_loader')

    pub = rospy.Publisher('demo_voxel_grid', numpy_msg(Float32MultiArray), queue_size=10)

    # base_path = rospkg.RosPack().get_path('mps_shape_completion')
    base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/003_cracker_box/gt/"
    base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/025_mug/gt/"
    # base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/025_mug/non_occupy/"

    
    files = [f for f in os.listdir(base_path)]
    files.sort()

    for filename in files:
        if rospy.is_shutdown():
            break
            
        with open(os.path.join(base_path,filename)) as f:
            vox = binvox_rw.read_as_3d_array(f).data

        
        pub.publish(vox_to_msg(vox))    
        rospy.sleep(0.5)


if __name__ == '__main__':
    demo()
