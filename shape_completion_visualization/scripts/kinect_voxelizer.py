#!/usr/bin/env python
from __future__ import print_function

import rospy
from sensor_msgs.msg import PointCloud2
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from shape_completion_training.voxelgrid import conversions
from rviz_voxelgrid_visuals import conversions as msg_conversions
import ros_numpy
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils import data_tools
import tensorflow as tf
from shape_completion_training.model import utils
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher
import rospkg
import pickle
import numpy as np

# target_frame = "mocap_world"
target_frame = "victor_root"

# Mug fitting
scale = 0.003
origin = (2.446 - scale * 32, -0.384 - scale * 32, 0.86 - scale * 32)
x_bounds = (0, 64)
# x_bounds = (20, 43)
y_bounds = (20, 46)
z_bounds = (0, 64)

# YCB fitting
# scale = 0.004
# origin = (2.45, -0.48, 0.75)
# x_bounds = (28, 37)
# y_bounds = (15, 64)
# z_bounds = (0, 64)


trial = "NormalizingAE/July_02_15-15-06_ede2472d34"


# trial = "NormalizingAE_YCB/July_24_11-21-46_f2aea4d768"


# Networks were trained with a different axis, a different notion of "up"
def transform_to_network(vg):
    vg = tf.reverse(vg, axis=[2])
    vg = tf.transpose(vg, perm=[0, 1, 3, 2, 4])
    return vg


def transform_from_network(vg):
    vg = tf.transpose(vg, perm=[0, 1, 3, 2, 4])
    vg = tf.reverse(vg, axis=[2])
    return vg


def swap_y_z_elem(elem):
    for k in ['known_occ', 'known_free']:
        elem[k] = transform_to_network(elem[k])
    return elem


def swap_y_z_inference(elem):
    for k in ['predicted_occ']:
        elem[k] = transform_from_network(elem[k])
    return elem


def crop_vg(vg):
    # vg = tf.convert_to_tensor(vg)
    vg_crop = vg.copy()
    vg_crop[z_bounds[0]:z_bounds[1], x_bounds[0]:x_bounds[1], y_bounds[0]:y_bounds[1], :] = 0
    return vg - vg_crop


def to_msg(voxelgrid):
    return msg_conversions.vox_to_occupancy_stamped(voxelgrid,
                                                    dim=voxelgrid.shape[1],
                                                    scale=scale,
                                                    frame_id=target_frame,
                                                    origin=origin)


def publish_simulated_mug():
    """
    Publish mug from simulated data, to verify axes of simulated data match real data
    :return:
    """
    path = rospkg.RosPack().get_path('shape_completion_visualization') + "/example_data/"
    # path += "front_left_mug.pkl"
    path += "mug_hidden_handle.pkl"

    with open(path) as f:
        mug = pickle.load(f)
    mug = transform_from_network(mug)
    VG_PUB.publish('predicted_occ', mug)


def infer(elem):
    elem = utils.add_batch_to_dict(elem)
    elem = swap_y_z_elem(elem)

    for i in range(5):
        inference = model_runner.model(elem)
        inference = swap_y_z_inference(inference)
        VG_PUB.publish_elem_cautious(inference)
        rospy.sleep(0.2)


def voxelize_point_cloud(pts):
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts)
    vg = conversions.pointcloud_to_voxelgrid(xyz_array, scale=scale, origin=origin, add_trailing_dim=True,
                                             add_leading_dim=False)
    vg = crop_vg(vg)
    # vg = np.swapaxes(vg, 1,2)
    elem = {'known_occ': vg}
    ko, kf = data_tools.simulate_2_5D_input(vg)

    if "YCB" in trial:
        # ko, kf = data_tools.simulate_slit_occlusion(ko, kf, x_bounds[0], x_bounds[1])
        kf[:, 0:x_bounds[0], :, 0] = 0
        kf[:, x_bounds[1]:, :, 0] = 0

    elem['known_free'] = kf
    return elem


def publish_elem(elem):
    VG_PUB.publish_elem_cautious(elem)
    # known_occ_pub.publish(to_msg(elem['known_occ']))
    # known_free_pub.publish(to_msg(elem['known_free']))
    # if 'predicted_occ' in elem:
    #     infer_pub.publish(to_msg(elem['predicted_occ']))
    # else:
    #     print("no prediction to publish")


def kinect_callback(msg):
    print("Point cloud received")

    timeout = 1.0
    try:
        trans = tf_buffer.lookup_transform(target_frame, msg.header.frame_id, msg.header.stamp, rospy.Duration(timeout))
    except tf2.LookupException as ex:
        rospy.logwarn(ex)
        return
    except tf2.ExtrapolationException as ex:
        rospy.logwarn(ex)
        return
    cloud_out = do_transform_cloud(msg, trans)

    points = []
    # for point in sensor_msgs.point_cloud2.read_points(cloud_out, skip_nans=True):
    #     points.append(point)

    elem = voxelize_point_cloud(cloud_out)
    publish_elem(elem)
    # publish_simulated_mug()
    infer(elem)

    # msg = to_msg(vg)
    print("Made a cloud")


if __name__ == "__main__":
    model_runner = ModelRunner(training=False, trial_path=trial)
    rospy.init_node("kinect_voxelizer")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    kinect_sub = rospy.Subscriber("/kinect2_victor_head/sd/points", PointCloud2, kinect_callback)

    VG_PUB = VoxelgridPublisher(frame=target_frame, scale=scale, origin=origin)

    # recomputed_pub = rospy.Publisher("/recomputed_cloud", PointCloud2)
    # known_occ_pub = rospy.Publisher("/kinect_voxels", VoxelgridStamped, queue_size=1)
    # known_free_pub = rospy.Publisher("/known_free", VoxelgridStamped, queue_size=1)
    # infer_pub = rospy.Publisher("/predicted_occ", VoxelgridStamped, queue_size=1)
    rospy.spin()
