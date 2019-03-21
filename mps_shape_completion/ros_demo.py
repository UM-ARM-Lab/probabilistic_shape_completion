#!/usr/bin/env python

from std_msgs.msg import ByteMultiArray, MultiArrayDimension
from rospy.numpy_msg import numpy_msg

import rospy

import numpy as np
import time
import binvox_rw
import rospkg

DIM = 64
base_path = ''


def callback(msg):
    global base_path
    arr = np.reshape(msg.data, tuple(d.size for d in msg.layout.dim))
    print rospy.get_name(), "I heard %s"%str(arr)
    occ = arr > 0
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
    
    msg = ByteMultiArray()
    msg.data = (occ.astype(int) - non.astype(int)).flatten().tolist()
    msg.layout.dim.append(MultiArrayDimension(label='x', size=DIM, stride=DIM*DIM*DIM))
    msg.layout.dim.append(MultiArrayDimension(label='y', size=DIM, stride=DIM*DIM))
    msg.layout.dim.append(MultiArrayDimension(label='z', size=DIM, stride=DIM))

    rospy.init_node('shape_demo_loader')
    pub = rospy.Publisher('local_occupancy', numpy_msg(ByteMultiArray), queue_size=10)
    rospy.Subscriber("local_occupancy_predicted", numpy_msg(ByteMultiArray), callback)

    time.sleep(1)
    pub.publish(msg)
    rospy.spin()


if __name__ == '__main__':
    demo()
