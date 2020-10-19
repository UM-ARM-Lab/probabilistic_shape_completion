#!/usr/bin/env python
from __future__ import print_function

import rospy

import tensorflow as tf

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import CompressedImage, CameraInfo
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

import tf2_ros
import tf2_py as tf2
from shape_completion_training.voxelgrid import conversions
from rviz_voxelgrid_visuals import conversions as msg_conversions
import ros_numpy

from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils import data_tools
from shape_completion_training.model import utils
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher
import rospkg
import pickle
import numpy as np
import message_filters
import image_geometry
import shape_completion_robot_demo.utils as demo_utils
from arc_utilities.ros_helpers import Xbox
from amazon_ros_speech import talker
from arm_video_recorder.srv import TriggerVideoRecording, TriggerVideoRecordingRequest

from object_segmentation import object_segmentations as obseg

# target_frame = "mocap_world"
target_frame = "victor_root"

# talker.mute()

# Mug fitting
scale = 0.003
origin = (2.446 - scale * 32, -0.384 - scale * 32, 0.86 - scale * 32)
x_bounds = (0, 64)
# x_bounds = (20, 43)
y_bounds = (17, 46)
z_bounds = (0, 64)

# CAMERA_MODEL = None

# YCB fitting
# scale = 0.004
# origin = (2.45, -0.48, 0.75)
# x_bounds = (28, 37)
# y_bounds = (15, 64)
# z_bounds = (0, 64)


trial = "NormalizingAE/July_02_15-15-06_ede2472d34"
# trial = "NormalizingAE_noise/August_03_13-44-05_8c8337b208"
# trial = "NormalizingAE/September_10_21-15-32_f87bdf38d4"
# trial = "VAE_GAN/September_12_15-08-29_f87bdf38d4"
# trial = "3D_rec_gan/September_12_15-47-07_f87bdf38d4"


# trial = "NormalizingAE_YCB/July_24_11-21-46_f2aea4d768"


ALREADY_PROCESSING = False
xbox = None


def should_grasp_check():
    return xbox.get_button("A")


def get_grasp_point(inference):
    pred_pts = conversions.voxelgrid_to_pointcloud(inference['predicted_occ'] > 0.5,
                                                   scale=scale, origin=(0, 0, 0))
    if len(pred_pts) == 0:
        return None

    offset = VG_PUB.origin
    centroid = np.mean(pred_pts, axis=0) + offset
    furthest_forward = pred_pts[np.argmin(pred_pts[:, 0]), :] + offset
    furthest_back = pred_pts[np.argmax(pred_pts[:, 0]), :] + offset

    return furthest_back


def is_grasp_point_valid(point):
    # return point[1] > 0.228
    return point[0] > 0.6


def publish_grasp_point(point, clear=False):
    if point is None:
        point = [0,0,0]

    m = Marker()
    m.ns = "Grasp Marker"
    m.header.frame_id = "victor_root"
    m.pose.position.x, m.pose.position.y, m.pose.position.z = point
    # m.pose.position.z = point[2] + 0.42
    m.type = Marker.SPHERE
    m.color.a = 1
    m.color.r = 0
    if not is_grasp_point_valid(point):
        m.color.r = 1
    m.color.g = 0
    m.color.b = 0
    if clear:
        m.scale.x = 0.001
        m.scale.y = 0.001
        m.scale.z = 0.001
    else:
        m.scale.x = 0.02
        m.scale.y = 0.05
        m.scale.z = 0.02

    grasp_marker_pub.publish(m)


def grasp(elem):
    if not should_grasp_check():
        return

    inferences = []
    for i in range(20):
        inferences.append(infer(elem))

    talker.say("Here are the completions")
    while not rospy.is_shutdown() and not xbox.get_button("A"):
        all_valid_grasp_points = []
        for inference in inferences:
            VG_PUB.publish_elem_cautious(inference)
            grasp_point = get_grasp_point(inference)
            publish_grasp_point(grasp_point)
            if is_grasp_point_valid(grasp_point):
                all_valid_grasp_points.append(grasp_point)
            rospy.sleep(0.1)
        xbox.wait_for_buttons(["A", "B"])
    rospy.sleep(1)

    for inference in inferences:
        VG_PUB.publish_elem_cautious(inference)
        publish_grasp_point(get_grasp_point(inference))
        b = xbox.wait_for_buttons(["A", "B"])
        rospy.sleep(0.2)
        if b == "B":
            talker.say("skipping")
            continue
        if b == "A":
            talker.say("selected")
            break


    if len(all_valid_grasp_points) == 0:
        talker.say("No valid grasp points")
        return

    grasp_point = np.mean(all_valid_grasp_points, axis=0)

    execute_grasp(grasp_point)


def execute_grasp(point):
    video = rospy.ServiceProxy("/video_recorder", TriggerVideoRecording)
    req = TriggerVideoRecordingRequest()

    req.filename = "mug_grasp_{}.mp4".format(demo_utils.get_datetime_str())
    req.timeout_in_sec = 60.0
    req.record = True
    video(req)

    publish_grasp_point(point)
    talker.say("Confirm grasp?")
    rospy.sleep(0.5)



    if "B" == xbox.wait_for_buttons(["A", "B"], message=False):
        talker.say("Skipping")
    else:
        talker.say("Grasping")
        rospy.sleep(0.5)

        pt = Point()
        pt.x, pt.y, pt.z = point
        print("Grasping {}".format(pt))
        grasp_pub.publish(pt)



    publish_grasp_point(None, clear=True)
    xbox.wait_for_buttons("A")
    req.record=False
    video(req)


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

    # return

    # for i in range(50):
    #     inference = model_runner.model(elem)
    #     inference = swap_y_z_inference(inference)
    #     VG_PUB.publish_elem_cautious(inference)
    #     rospy.sleep(0.2)
    # return

    inference = model_runner.model(elem)
    inference = swap_y_z_inference(inference)
    VG_PUB.publish_elem_cautious(inference)
    return inference

    # rospy.sleep(0.2)


def voxelize_point_cloud(pts):
    global origin
    global VG_PUB
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(pts)

    if len(xyz_array) == 0:
        origin = (0, 0, 0)
    else:
        origin = np.mean(xyz_array, axis=0) - np.array([scale * 32 - .04, scale * 32, scale * 32])
    VG_PUB.origin = origin
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

    elem['known_occ'] = ko
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


def filter_pointcloud(img_mask_msg, img_rect_msg, depth_rect_msg):
    if camera_model.camera_model is None:
        print("Waiting for camera model")
        return
    img_mask = demo_utils.decompress_img(img_mask_msg)
    img_rect = demo_utils.decompress_img(img_rect_msg)
    depth_rect = demo_utils.decompress_depth(depth_rect_msg)

    # prediction = segmenter.run_inference_for_single_image(img_rect)
    # img_rect_msg.data = obseg.compress_img(prediction)
    # marked_pub.publish(img_rect_msg)

    pts = demo_utils.convert_masked_depth_img_to_pointcloud(depth_rect, img_rect, img_mask,
                                                            camera_model.camera_model)

    # pts = demo_utils.convert_depth_img_to_pointcloud(depth_rect, img_rect,
    #                                                  camera_model.camera_model)
    # pt_msg = point_cloud2.create_cloud_xyz32(depth_rect_msg.header, pts)
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
              PointField('y', 4, PointField.FLOAT32, 1),
              PointField('z', 8, PointField.FLOAT32, 1),
              # PointField('rgb', 12, PointField.UINT32, 1),
              PointField('rgba', 12, PointField.UINT32, 1),
              ]
    header = Header()
    header.frame_id = img_rect_msg.header.frame_id
    pt_msg = point_cloud2.create_cloud(header, fields, pts)

    cloud_pub.publish(pt_msg)

    return pt_msg


def kinect_callback(img_mask_msg, img_rect_msg, depth_rect_msg):
    global ALREADY_PROCESSING
    if ALREADY_PROCESSING:
        return
    ALREADY_PROCESSING = True
    # def kinect_callback(msg):
    print("Point cloud received")

    pt_msg = filter_pointcloud(img_mask_msg, img_rect_msg, depth_rect_msg)

    print("Published cloud")

    # return

    timeout = 1.0
    try:
        trans = tf_buffer.lookup_transform(target_frame, pt_msg.header.frame_id,
                                           pt_msg.header.stamp, rospy.Duration(timeout))
    except tf2.LookupException as ex:
        rospy.logwarn(ex)
        return
    except tf2.ExtrapolationException as ex:
        rospy.logwarn(ex)
        return
    cloud_out = do_transform_cloud(pt_msg, trans)

    points = []
    # for point in sensor_msgs.point_cloud2.read_points(cloud_out, skip_nans=True):
    #     points.append(point)

    elem = voxelize_point_cloud(cloud_out)
    publish_elem(elem)
    # publish_simulated_mug()
    inference = infer(elem)

    grasp(elem)

    ALREADY_PROCESSING = False


if __name__ == "__main__":
    rospy.init_node("kinect_shape_completion_demo")
    model_runner = ModelRunner(training=False, trial_path=trial)

    # segmenter = obseg.Segmenter(gpu=None)
    # marked_pub = rospy.Publisher("/marked_image/compressed", CompressedImage, queue_size=1)

    cloud_pub = rospy.Publisher("/republished_pointcloud", PointCloud2, queue_size=1)
    debug_cloud_pub = rospy.Publisher("debug_pointcloud", PointCloud2, queue_size=1)

    grasp_marker_pub = rospy.Publisher("grasp_marker", Marker, queue_size=1)
    grasp_pub = rospy.Publisher("/grasp_point_command", Point, queue_size=1)
    xbox = Xbox(xpad=False)
    talker.init()

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    last_update = rospy.get_rostime()
    # camera_info_sub = rospy.Subscriber("/kinect2_victor_head/hd/camera_info", CameraInfo, camera_info_callback)
    camera_model = demo_utils.CameraModel("/kinect2_victor_head/qhd/camera_info")

    # kinect_sub = rospy.Subscriber("/kinect2_victor_head/sd/points", PointCloud2, kinect_callback)
    # image_sub = message_filters.Subscriber("/kinect2_victor_head/hd/image_color/compressed", CompressedImage)
    # image_rect_sub = message_filters.Subscriber("/kinect2_victor_head/hd/image_color_rect/compressed", CompressedImage)
    # depth_image_sub = message_filters.Subscriber("/kinect2_victor_head/hd/image_depth_rect/compressed", CompressedImage)

    # time_sync = message_filters.TimeSynchronizer([image_sub, image_rect_sub, depth_image_sub], 10)

    # image_sub = message_filters.Subscriber("/kinect2_victor_head/hd/image_color/compressed", CompressedImage)
    image_mask_sub = message_filters.Subscriber("/segmentation_mask/compressed", CompressedImage)
    image_rect_sub = message_filters.Subscriber("/kinect2_victor_head/qhd/image_color_rect/compressed", CompressedImage)
    depth_image_sub = message_filters.Subscriber("/kinect2_victor_head/qhd/image_depth_rect/compressed",
                                                 CompressedImage)

    time_sync = message_filters.TimeSynchronizer([image_mask_sub, image_rect_sub, depth_image_sub], 10)
    time_sync.registerCallback(kinect_callback)

    VG_PUB = VoxelgridPublisher(frame=target_frame, scale=scale, origin=origin)

    # recomputed_pub = rospy.Publisher("/recomputed_cloud", PointCloud2)
    # known_occ_pub = rospy.Publisher("/kinect_voxels", VoxelgridStamped, queue_size=1)
    # known_free_pub = rospy.Publisher("/known_free", VoxelgridStamped, queue_size=1)
    # infer_pub = rospy.Publisher("/predicted_occ", VoxelgridStamped, queue_size=1)
    rospy.spin()
