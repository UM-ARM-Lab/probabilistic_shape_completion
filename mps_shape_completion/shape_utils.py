#!/usr/bin/env python

from std_msgs.msg import Float32MultiArray, MultiArrayDimension
import numpy as np

DIM = 64


def vox_to_msg(voxel_grid):
    out_msg = Float32MultiArray()
    out_msg.data = voxel_grid.astype(np.float32).flatten().tolist()
    out_msg.layout.dim.append(MultiArrayDimension(label='x', size=DIM, stride=DIM*DIM*DIM))
    out_msg.layout.dim.append(MultiArrayDimension(label='y', size=DIM, stride=DIM*DIM))
    out_msg.layout.dim.append(MultiArrayDimension(label='z', size=DIM, stride=DIM))
    return out_msg


def msg_to_vox(msg):
    return np.reshape(msg.data, tuple(d.size for d in msg.layout.dim))
