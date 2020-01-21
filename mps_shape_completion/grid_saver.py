#!/usr/bin/env python

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import rospy

import numpy as np
import time
from datetime import datetime
import binvox_rw

import roslib
PKG = 'mps_voxels'
roslib.load_manifest(PKG)

DIM = 64


def callback(msg, args):
    file_prefix = args
    arr = np.reshape(msg.data, tuple(d.size for d in msg.layout.dim))
    occ = arr > 0
    non = arr < 0
    # Save to file for demo
    timestr = datetime.utcnow().strftime("%Y-%m-%d-%H-%M-%S-%f")[:-3]

    vox = binvox_rw.Voxels(occ, [64, 64, 64], [0, 0, 0], 1, 'xyz')
    filename = file_prefix + '_occupied_' + timestr + '.binvox'
    with open(filename, 'wb') as f:
        vox.write(f)
        print('Output saved to ' + filename + '.')

    vox = binvox_rw.Voxels(non, [64, 64, 64], [0, 0, 0], 1, 'xyz')
    filename = file_prefix + '_unoccupied_' + timestr + '.binvox'
    with open(filename, 'wb') as f:
        vox.write(f)
        print('Output saved to ' + filename + '.')
    

def listener():
    """
    Write ROS data to file
    """

    rospy.init_node('voxel_grid_saver', anonymous=True)
    default_topic_name = "local_occupancy_predicted"
    file_prefix = rospy.resolve_name(default_topic_name)[1:]
    sub = rospy.Subscriber(default_topic_name, numpy_msg(Float32MultiArray), callback, callback_args=file_prefix)

    rospy.spin()


if __name__ == '__main__':
    listener()
