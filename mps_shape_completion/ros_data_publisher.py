#!/usr/bin/env python

"""
Cycles through and publishes data for visualization
"""

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from mps_shape_completion_msgs.msg import OccupancyStamped
from rospy.numpy_msg import numpy_msg

import rospy
from shape_utils import vox_to_msg

from shape_complete import ShapeCompleter

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

    model_path = rospkg.RosPack().get_path('mps_shape_completion') + "/train_mod/"
    sc = ShapeCompleter(model_path, verbose=True)


    gt_pub = rospy.Publisher('gt_voxel_grid_stamped', OccupancyStamped, queue_size=10)
    occ_input_pub = rospy.Publisher('occ_input_voxel_grid_stamped', OccupancyStamped, queue_size=10)
    # completion_raw_pub = rospy.Publisher('predicted_voxel_grid', numpy_msg(Float32MultiArray), queue_size=10)
    completion_pub = rospy.Publisher('predicted_voxel_grid_stamped', OccupancyStamped, queue_size=10)

    # ycb_obj = "025_mug"
    # ycb_obj = "019_pitcher_base"
    # ycb_obj = "007_tuna_fish_can"
    ycb_obj = "shapenet_002_mug"
    
    # base_path = rospkg.RosPack().get_path('mps_shape_completion')
    base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/"
    gt_path = base_path + ycb_obj + "/gt/"
    occ_path = base_path + ycb_obj + "/train_x_occ/"
    non_occ_path = base_path + ycb_obj + "/non_occupy/"
    # base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/025_mug/non_occupy/"

    
    files = [f for f in os.listdir(occ_path)]
    files.sort()

    for filename in files:
        if rospy.is_shutdown():
            break

        prefix = filename.split("occupy")[0]
        print(prefix)
            
        with open(os.path.join(gt_path,prefix + "gt.binvox")) as f:
            gt_vox = binvox_rw.read_as_3d_array(f).data


        
        with open(os.path.join(occ_path,prefix + "occupy.binvox")) as f:
            occ_vox = binvox_rw.read_as_3d_array(f).data


        with open(os.path.join(non_occ_path,prefix + "non_occupy.binvox")) as f:
            non_occ_vox = binvox_rw.read_as_3d_array(f).data

        out = sc.complete(occ=occ_vox, non=non_occ_vox, verbose=False, save=False)

        gt_msg = OccupancyStamped()
        gt_msg.header.frame_id = "base_frame"
        gt_msg.occupancy = vox_to_msg(gt_vox)
        gt_msg.scale = 0.01
        gt_pub.publish(gt_msg)

        input_msg = OccupancyStamped()
        input_msg.header.frame_id = "base_frame"
        input_msg.occupancy = vox_to_msg(occ_vox)
        input_msg.scale = 0.01
        occ_input_pub.publish(input_msg)
                
        # occ_input_pub.publish(vox_to_msg(occ_vox))

        # completion_raw_pub.publish(vox_to_msg(out))

        c_msg = OccupancyStamped()
        c_msg.header.frame_id = "base_frame"
        c_msg.occupancy = vox_to_msg(out)
        c_msg.scale = 0.01
        completion_pub.publish(c_msg)

        
        rospy.sleep(0.5)


if __name__ == '__main__':
    demo()
