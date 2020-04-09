#! /usr/bin/env python
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np

import IPython




def iou(voxelgrid_1, voxelgrid_2, threshold=0.5):
    """
    calculates the intersection over union between two voxelgrids of the same size.
    Both voxelgrids are treated as binary voxel grids using the threshold
    """
    v1 = tf.cast(voxelgrid_1 > threshold, tf.float32)
    v2 = tf.cast(voxelgrid_2 > threshold, tf.float32)

    intersection = tf.reduce_sum(tf.cast((v1 + v2) > 1.5, tf.float32))
    union = tf.reduce_sum(tf.cast((v1+v2) > 0.5, tf.float32))
    return intersection / union
