#! /usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import numpy as np


def iou(voxelgrid_1, voxelgrid_2, threshold=0.5):
    """
    Calculates the intersection over union between two voxelgrids of the same size.
    Both voxelgrids are treated as binary voxel grids using the threshold
    """
    v1 = tf.cast(voxelgrid_1 > threshold, tf.float32)
    v2 = tf.cast(voxelgrid_2 > threshold, tf.float32)

    intersection = tf.reduce_sum(tf.cast((v1 + v2) > 1.5, tf.float32))
    union = tf.reduce_sum(tf.cast((v1+v2) > 0.5, tf.float32))
    return intersection / union


def p_correct(estimate_voxelgrid, gt_voxelgrid):
    """
    Calculates the p(gt_voxelgrid | estimate_voxelgrid), assuming 
    estimate_voxelgrid contains a probability (0-1) of each voxel being occupied
    gt_voxelgrid contains the occupancy {0,1.0}
    """
    gt = tf.cast(gt_voxelgrid > 0.5, tf.float32)
    p_voxel_correct = 1.0 - (gt - estimate_voxelgrid)
    p_correct = tf.exp(tf.reduce_sum(tf.math.log(p_voxel_correct)))
    return p_correct


def p_correct_geometric_mean(estimate_voxelgrid, gt_voxelgrid):
    """
    Calculates the geometric mean p(gt_voxelgrid | estimate_voxelgrid), assuming 
    estimate_voxelgrid contains a probability (0-1) of each voxel being occupied
    gt_voxelgrid contains the occupancy {0,1.0}
    """
    gt = tf.cast(gt_voxelgrid > 0.5, tf.float32)
    p_voxel_correct = 1.0 - tf.math.abs(gt - estimate_voxelgrid)
    num_elements = tf.cast(tf.size(p_voxel_correct), tf.float32)
    p_correct = tf.exp(tf.reduce_sum(tf.math.log(p_voxel_correct))/num_elements)
    return p_correct


def highest_match(test_vg, vg_list, metric=iou):
    """
    Returns the index and element of vg_list that maximizes metric(test_vg, element)
    @param test_vg: voxelgrid
    @param vg_list: list of voxelgrids
    @param metric: metric function of two voxelgrids
    @return: (index, element)

    TODO:
    """

    best_ind = None
    best_elem = None
    best_val = -np.inf

    for i, elem in enumerate(vg_list):
        val = metric(test_vg, elem)
        if val > best_val:
            best_ind = i
            best_elem = elem
            best_val = val
    return best_ind, best_elem


