#!/usr/bin/env python

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import rospy
from shape_utils import vox_to_msg

import numpy as np
import time
import binvox_rw
import rospkg

base_path = ''


def callback(msg):
    global base_path
    arr = np.reshape(msg.data, tuple(d.size for d in msg.layout.dim))
    print rospy.get_name(), "I heard %s"%str(arr)
    occ = arr > 0.5
    # Save to file for demo
    vox = binvox_rw.Voxels(occ, occ.shape, [0,0,0], 1, 'xyz')
    with open(base_path + '/demo/output.binvox','wb') as f:
        vox.write(f)
        print('Output saved to demo/output.binvox.')
    
    rospy.signal_shutdown("Got result.")


def demo():
    '''
    Publish sample data to ROS
    '''
    global base_path

    base_path = rospkg.RosPack().get_path('mps_shape_completion')

    # Read demo binvox as (64*64*64) array
    with open(base_path + '/demo/occupy.binvox', 'rb') as f:
        occ = binvox_rw.read_as_3d_array(f).data
    with open(base_path + '/demo/non_occupy.binvox', 'rb') as f:
        non = binvox_rw.read_as_3d_array(f).data
    


    rospy.init_node('shape_demo_loader')

    rospy.wait_for_service('complete_shape')

    pub = rospy.Publisher('local_occupancy', numpy_msg(Float32MultiArray), queue_size=10)
    rospy.Subscriber("local_occupancy_predicted", numpy_msg(Float32MultiArray), callback)

    time.sleep(1)
    print("Requesting shape completion")
    pub.publish(vox_to_msg(occ))
    rospy.spin()


if __name__ == '__main__':
    demo()
