#!/usr/bin/env python
from __future__ import print_function


import rospy
import tf2_ros
import tf_conversions
import geometry_msgs.msg
from mps_shape_completion_msgs.msg import OccupancyStamped
from mps_shape_completion_visualization import conversions


import numpy as np

import sys
import os

sc_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(sc_path)
from shape_completion_training.model.network import AutoEncoderWrapper
from shape_completion_training.model import data_tools
from shape_completion_training.model import obj_tools
from shape_completion_training import binvox_rw
import IPython



# DIM = 64

gt_pub = None
known_occ_pub = None
known_free_pub = None
completion_pub = None
mismatch_pub = None



def to_msg(voxel_grid):
    
    return conversions.vox_to_occupancy_stamped(voxel_grid,
                                                dim = voxel_grid.shape[1],
                                                scale=0.01,
                                                frame_id = "object")


def view_single_binvox():
    gt_pub = rospy.Publisher('gt_voxel_grid', OccupancyStamped, queue_size=1)
    fp = "/home/bsaund/catkin_ws/src/mps_shape_completion/shape_completion_training/data/ShapeNetCore.v2_augmented/03797390"
    # fp = os.path.join(fp, "a1d293f5cc20d01ad7f470ee20dce9e0")
    fp = os.path.join(fp, "214dbcace712e49de195a69ef7c885a4")
    fp = os.path.join(fp, "models")
    # fn = "model_normalized.obj_64.binvox"
    fn = "model_normalized.binvox"
    # fn = "model_normalized.solid.binvox"

    fp = os.path.join(fp, fn)
    
    with open(fp) as f:
        gt_vox = binvox_rw.read_as_3d_array(f).data

    print("Publishing single binvox {}".format(fp))
    gt_pub.publish(to_msg(gt_vox))
    rospy.sleep(10)


def publish_test_img():

    mug_fp = "/home/bsaund/catkin_ws/src/mps_shape_completion/shape_completion_training/data/ShapeNetCore.v2_augmented/03797390/"


    shapes = os.listdir(mug_fp)
    shapes = ['214dbcace712e49de195a69ef7c885a4']
    
    for shape in shapes:
        shape_fp = os.path.join(mug_fp, shape, "models")
        print ("Displaying {}".format(shape))

        gt_fp = os.path.join(shape_fp, "model_normalized.solid.binvox")

        with open(gt_fp) as f:
            gt_vox = binvox_rw.read_as_3d_array(f).data
        gt_pub.publish(to_msg(gt_vox))
        rospy.sleep(1)

        augs = [f for f in os.listdir(shape_fp) if f.startswith("model_augmented")]
        augs.sort()
        # IPython.embed()
        for aug in augs:

            if rospy.is_shutdown():
                return
        
            with open(os.path.join(shape_fp, aug)) as f:
                ko_vox = binvox_rw.read_as_3d_array(f).data
    
            known_occ_pub.publish(to_msg(ko_vox))

            print("Publishing {}".format(aug))
            rospy.sleep(.5)


def publish_elem(elem):
    gt_pub.publish(to_msg(elem["gt_occ"].numpy()))
    known_occ_pub.publish(to_msg(elem["known_occ"].numpy()))
    known_free_pub.publish(to_msg(elem['known_free'].numpy()))
    sys.stdout.write('\033[2K\033[1G')
    print("Category: {}, id: {}, aug: {}".format(elem['shape_category'].numpy(),
                                                 elem['id'].numpy(),
                                                 elem['augmentation'].numpy()), end="")
    sys.stdout.flush()


def publish_shapenet_tfrecords():
    data = data_tools.load_shapenet([data_tools.shape_map["mug"]])
    data = data_tools.simulate_input(data, 5, 5, 5)

    # print(sum(1 for _ in data))

    print("")
    
    for elem in data.batch(1):
        if rospy.is_shutdown():
            return
        publish_elem(elem)
        rospy.sleep(0.5)


def publish_completion():

    print("Loading...")
    data = data_tools.load_shapenet([data_tools.shape_map["mug"]])
    data = data_tools.simulate_input(data, 0, 0, 0)
    
    model = AutoEncoderWrapper()
    # model.restore()
    # model.evaluate(data)
    print("")

    for elem in data:

        if rospy.is_shutdown():
            return
        
        sys.stdout.write('\033[2K\033[1G')
        print("Category: {}, id: {}".format(elem['shape_category'].numpy(),
                                            elem['id'].numpy()), end="")
        sys.stdout.flush()
        dim = elem['gt_occ'].shape[0]


        elem_expanded = {}
        for k in elem.keys():
            elem_expanded[k] = np.expand_dims(elem[k].numpy(), axis=0)

        inference = model.model(elem_expanded)['predicted_occ'].numpy()
        gt_pub.publish(to_msg(elem['gt_occ'].numpy()))
        known_occ_pub.publish(to_msg(elem_expanded['known_occ']))
        known_free_pub.publish(to_msg(elem_expanded['known_free']))
        completion_pub.publish(to_msg(inference))

        mismatch = np.abs(elem['gt_occ'].numpy() - inference)
        mismatch_pub.publish(to_msg(mismatch))
        # IPython.embed()

        rospy.sleep(0.5)


def layer_by_layer():
    print("Loading...")
    data = data_tools.load_shapenet([data_tools.shape_map["mug"]])
    
    model = AutoEncoderWrapper()
    model.restore()

    i = 0

    for elem in data:
        i += 1
        if i<8:
            continue

        if rospy.is_shutdown():
            return
        # IPython.embed()
        
        print("Publishing")
        dim = elem['gt_occ'].shape[0]

        elem_expanded = {}
        for k in elem.keys():
            elem_expanded[k] = np.expand_dims(elem[k].numpy(), axis=0)

        elem_expanded['known_occ'][:,:,32:] = 0.0
        elem_expanded['known_free'][:,:,32:] = 0.0
        
        d = 31
        inference = elem_expanded['known_occ'] * 0.0
        while d < 64:
            print("Layer {}".format(d))
            ko = elem_expanded['known_occ']
            # IPython.embed()
            ko[:,:,d] += inference[:,:,d]
            # IPython.embed()
            ko=ko.clip(min=0.0, max=1.0)
            elem_expanded['known_occ'] = ko
            # IPython.embed()
            

            inference = model.model.predict(elem_expanded)

            gt_pub.publish(to_msg(elem['gt'].numpy()))
            known_occ_pub.publish(to_msg(elem_expanded['known_occ']))
            known_free_pub.publish(to_msg(elem_expanded['known_free']))
            completion_pub.publish(to_msg(inference))

            # if d==31:
            #     IPython.embed()
            
            d += 1
            # rospy.sleep(0.5)
            
        # IPython.embed()
        rospy.sleep(2.0)

    

if __name__=="__main__":
    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    gt_pub = rospy.Publisher('gt_voxel_grid', OccupancyStamped, queue_size=1)
    known_occ_pub = rospy.Publisher('known_occ_voxel_grid', OccupancyStamped, queue_size=1)
    known_free_pub = rospy.Publisher('known_free_voxel_grid', OccupancyStamped, queue_size=1)
    completion_pub = rospy.Publisher('predicted_voxel_grid', OccupancyStamped, queue_size=1)
    mismatch_pub = rospy.Publisher('mismatch_voxel_grid', OccupancyStamped, queue_size=1)

    # Transform so shapes appear upright in rviz
    br = tf2_ros.TransformBroadcaster()
    t = geometry_msgs.msg.TransformStamped()
    t.header.stamp = rospy.Time.now()
    t.header.frame_id = "world"
    t.child_frame_id = "object"
    q = tf_conversions.transformations.quaternion_from_euler(1.57,0,0)
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]


    rospy.sleep(1)
    br.sendTransform(t)


    # view_single_binvox()
    # publish_test_img()
    publish_shapenet_tfrecords()
    # publish_completion()
    # layer_by_layer()
    # data_tools.write_shapenet_to_tfrecord()
    # data_tools.load_shapenet()
    
