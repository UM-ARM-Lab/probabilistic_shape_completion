import numpy as np
from numpy import sin, cos
import tensorflow as tf


def strip_extra_dims(vg):
    """
    strips the leading and trailing dimensions from the voxelgrid
    @param vg:
    @return: stripped voxelgrid, has_leading_dim, has_ending_dim
    """
    return np.squeeze(vg), vg.shape[0] == 1, vg.shape[-1] == 1


def add_extra_dims(vg, add_leading_dim, add_trailing_dim):
    if add_leading_dim:
        vg = np.expand_dims(vg, 0)
    if add_trailing_dim:
        vg = np.expand_dims(vg, -1)
    return vg


def get_format(vg):
    return vg.shape[0] == 1, vg.shape[-1] == 1


def format_voxelgrid(voxelgrid, leading_dim, trailing_dim):
    squeeze = lambda x: np.squeeze(x)
    expand_dims = lambda x, axis: np.expand_dims(x, axis)
    if tf.is_tensor(voxelgrid):
        squeeze = lambda x: tf.squeeze(x)
        expand_dims = lambda x, axis: tf.expand_dims(x, axis)

    voxelgrid = squeeze(voxelgrid)
    if leading_dim:
        voxelgrid = expand_dims(voxelgrid, 0)
    if trailing_dim:
        voxelgrid = expand_dims(voxelgrid, -1)
    return voxelgrid


def voxelgrid_to_pointcloud(voxelgrid, scale=1.0, origin=(0, 0, 0), threshold=0.5):
    """
    Converts a 3D voxelgrid into a 3D set of points for each voxel with value above threshold
    @param voxelgrid: (opt 1 x) X x Y x Z (opt x 1) voxelgrid
    @param scale:
    @param origin: origin in voxel coorindates
    @param threshold:
    @return:
    """
    pts = np.argwhere(np.squeeze(voxelgrid) > threshold)
    return (np.array(pts) - origin + 0.5) * scale
    # pts = tf.cast(tf.where(tf.squeeze(voxelgrid) > threshold), tf.float32)
    # return (pts - origin + 0.5) * scale


def pointcloud_to_voxelgrid(pointcloud, scale=1.0, origin=(0, 0, 0), shape=(64, 64, 64),
                            add_leading_dim=False, add_trailing_dim=False):
    """
    Converts a set of 3D points into a binary voxel grid
    @param pointcloud:
    @param scale: scale of the voxelgrid
    @param origin:
    @param shape:
    @param add_trailing_dim:
    @param add_leading_dim:
    @return:
    """
    vg = np.zeros(shape)
    if tf.is_tensor(pointcloud):
        pointcloud = pointcloud.numpy()
    s = (pointcloud / scale + origin).astype(int)
    vg[s[:, 0], s[:, 1], s[:, 2]] = 1.0
    return format_voxelgrid(vg, add_leading_dim, add_trailing_dim)


def transform_voxelgrid(vg, transform, scale=1.0, origin=(0, 0, 0)):
    """

    @param vg: voxelgrid
    @param transform: 4x4 homogeneous tramsform matrix
    @return:
    """
    vg, add_leading_dim, add_trailing_dim = strip_extra_dims(vg)

    pt_cloud = voxelgrid_to_pointcloud(vg, scale=scale, origin=origin)

    if pt_cloud.shape[0] == 0:
        return vg

    R = transform[0:3, 0:3]
    t = transform[0:3, 3]
    trans_cloud = np.dot(R, pt_cloud.transpose()).transpose() + t
    return pointcloud_to_voxelgrid(trans_cloud, scale=scale, origin=origin, shape=vg.shape,
                                   add_trailing_dim=add_trailing_dim,
                                   add_leading_dim=add_leading_dim)


def make_transform(thetas=(0, 0, 0), translation=(0, 0, 0)):
    rot_x = [[1, 0, 0],
             [0, cos(thetas[0]), -sin(thetas[0])],
             [0, sin(thetas[0]), cos(thetas[0])]]
    rot_y = [[cos(thetas[1]), 0, sin(thetas[1])],
             [0, 1, 0],
             [-sin(thetas[1]), 0, cos(thetas[1])]]
    rot_z = [[cos(thetas[2]), -sin(thetas[2]), 0],
             [sin(thetas[2]), cos(thetas[2]), 0],
             [0, 0, 1]]
    R = np.dot(rot_x, np.dot(rot_y, rot_z))
    T = np.block([[R, np.expand_dims(np.array(translation), 0).transpose()], [np.zeros(3), 1]])
    return T


def downsample(voxelgrid, kernel_size=2):
    if kernel_size == 1:
        return voxelgrid

    if not tf.is_tensor(voxelgrid):
        voxelgrid = tf.cast(voxelgrid, tf.float32)
    leading, trailing = get_format(voxelgrid)
    formatted = format_voxelgrid(
        tf.nn.max_pool(format_voxelgrid(voxelgrid, True, True), ksize=kernel_size, strides=kernel_size,
                       padding="VALID"),
        leading, trailing)
    return formatted
