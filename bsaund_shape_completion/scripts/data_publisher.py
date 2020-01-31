#!/usr/bin/env python

import rospy
from mps_shape_completion_msgs.msg import OccupancyStamped
from mps_shape_completion_visualization import conversions
import numpy as np

import sys
import os

sc_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(sc_path)
from shape_completion_training.model.network import AutoEncoderWrapper
from shape_completion_training.model import data_tools
import IPython

# DIM = 64


def to_msg(voxel_grid):
    
    return conversions.vox_to_occupancy_stamped(voxel_grid,
                                                dim = voxel_grid.shape[1],
                                                scale=0.01,
                                                frame_id = "base_frame")



def publish():

    print("Loading...")
    # data = data_tools.load_data(from_record=True)
    # data = data_tools.load_data(from_record=False)
    data = data_tools.load_shapenet([data_tools.shape_map["mug"]])
    # data = data.take(1).repeat(400)
    # data = data_tools.load_shapenet()


    print("Sharding...")
    # data = data.shard(100, 0)
    print("Shuffling...")
    data = data.shuffle(100)
    # data = data.skip(4000)
    # IPython.embed()
    
    model = AutoEncoderWrapper()
    # model.evaluate(data)
    model.restore()
    # model.evaluate(data)

    gt_pub = rospy.Publisher('gt_voxel_grid', OccupancyStamped, queue_size=10)
    known_occ_pub = rospy.Publisher('known_occ_voxel_grid', OccupancyStamped, queue_size=10)
    known_free_pub = rospy.Publisher('known_free_voxel_grid', OccupancyStamped, queue_size=10)
    completion_pub = rospy.Publisher('predicted_voxel_grid', OccupancyStamped, queue_size=10)


    for elem, _ in data:
        if rospy.is_shutdown():
            return
        # IPython.embed()
        
        print("Publishing")
        dim = elem['gt'].shape[0]
        gt_msg = to_msg(elem['gt'].numpy())
        # gt_msg = conversions.vox_to_occupancy_stamped(elem['gt'].numpy(), dim=dim, scale=0.01, frame_id="base_frame")
        gt_pub.publish(gt_msg)

        elem_expanded = {}
        for k in elem.keys():
            elem_expanded[k] = np.expand_dims(elem[k].numpy(), axis=0)

        ko = elem_expanded['known_occ']
        ko[:,:,32:] = 0.0
        elem_expanded['known_occ'] = ko
        elem_expanded['known_free'][:,:,32:] = 0.0


        known_occ_msg = to_msg(elem_expanded['known_occ'])
        known_occ_pub.publish(known_occ_msg)

        known_free_pub.publish(to_msg(elem_expanded['known_free']))

        inference = model.model.predict(elem_expanded)
        # completion_msg = conversions.vox_to_occupancy_stamped(inference,
        #                                                       dim=dim, scale=0.01, frame_id="base_frame")
        # IPython.embed()
        completion_msg = to_msg(inference)

        completion_pub.publish(completion_msg)
        

        
        rospy.sleep(0.5)

    

if __name__=="__main__":
    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")


    publish()
    # data_tools.write_shapenet_to_tfrecord()
    # data_tools.load_shapenet()
