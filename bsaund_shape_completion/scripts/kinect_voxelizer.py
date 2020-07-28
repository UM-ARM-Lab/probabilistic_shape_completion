#!/usr/bin/env python
from __future__ import print_function

import argparse
import rospy
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2
import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from shape_completion_training.voxelgrid import conversions
from mps_shape_completion_visualization import conversions as msg_conversions
import ros_numpy
from mps_shape_completion_msgs.msg import OccupancyStamped
from shape_completion_training.model.modelrunner import ModelRunner

# target_frame = "mocap_world"
target_frame = "victor_root"

scale = 0.01
origin = (0, -0.5, 0.5)


def to_msg(voxelgrid):
    return msg_conversions.vox_to_occupancy_stamped(voxelgrid,
                                                    dim=voxelgrid.shape[1],
                                                    scale=scale,
                                                    frame_id=target_frame,
                                                    origin=origin)


def voxelize_point_cloud(pts):
    vg = conversions.pointcloud_to_voxelgrid(pts, scale=scale, origin=origin)
    msg = to_msg(vg)
    vg_pub.publish(msg)

    # vg_pub.publish
    pass

def publish_kinect(vg):
    pass


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

    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(cloud_out)
    voxelize_point_cloud(cloud_out)

    # msg = to_msg(vg)
    print("Made a cloud")


if __name__ == "__main__":
    model_runner = ModelRunner(training=False, trial_path="NormalizingAE/July_02_15-15-06_ede2472d34")
    rospy.init_node("kinect_voxelizer")
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    kinect_sub = rospy.Subscriber("/kinect2_victor_head/sd/points", PointCloud2, kinect_callback)
    recomputed_pub = rospy.Publisher("/recomputed_cloud", PointCloud2)
    vg_pub = rospy.Publisher("/kinect_voxels", OccupancyStamped, queue_size=1)
    rospy.spin()
