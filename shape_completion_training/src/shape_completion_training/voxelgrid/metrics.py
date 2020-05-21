#! /usr/bin/env python
from __future__ import print_function
import tensorflow as tf
import numpy as np
from shape_completion_training.voxelgrid import conversions


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


def distance_matrix(pt_1, pt_2):
    """
    Computes the distance matrix from each point pt_1 to pt_2
    @param pt_1: tensor, [num_pts_1, pt_length]
    @param pt_2: tensor, [num_pts_2, pt_length]
    @return: distances [num_pts_1, num_pts_2] where distances[i,j] is the distance(pt_1[i], pt_1[j])
    """

    num_points_1, num_features = pt_1.shape
    num_points_2, num_features = pt_2.shape

    expanded_pt_2 = tf.tile(pt_2, (num_points_1, 1))
    expanded_pt_1 = tf.reshape(
        tf.tile(tf.expand_dims(pt_1, 1),
                (1, num_points_2, 1)),
        (-1, num_features))
    distances = tf.norm(expanded_pt_1 - expanded_pt_2, axis=1)
    distances = tf.reshape(distances, (num_points_1, num_points_2))
    return distances


def chamfer_distance_pointcloud(pt_1, pt_2):
    d = distance_matrix(pt_1, pt_2)
    d_ab = tf.reduce_mean(tf.reduce_min(d, axis=1))
    d_ba = tf.reduce_mean(tf.reduce_min(d, axis=0))
    return d_ab + d_ba


def chamfer_distance(vg1, vg2, scale, downsample=1):
    """
    Returns the chamfer distance between two voxelgrids
    """
    vg1 = conversions.downsample(vg1, downsample)
    vg2 = conversions.downsample(vg2, downsample)

    pt1 = conversions.voxelgrid_to_pointcloud(vg1, scale=scale)
    pt2 = conversions.voxelgrid_to_pointcloud(vg2, scale=scale)
    return chamfer_distance_pointcloud(pt1, pt2)
