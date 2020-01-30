#!/usr/bin/env python

import rospy
from mps_shape_completion_msgs.msg import OccupancyStamped
from mps_shape_completion_visualization import conversions
import numpy as np

import sys
import os

sc_path = os.path.join(os.path.dirname(__file__), "../../")
sys.path.append(sc_path)
from shape_completion_training.model.network import AutoEncoder
from shape_completion_training.model import data_tools
import IPython

# DIM = 64


def publish():
    # data = data_tools.load_data(from_record=True)
    
    # data = data_tools.load_data(from_record=False)
    data = data_tools.load_shapenet()
    # IPython.embed()
    
    # model = AutoEncoder()
    # model.evaluate(data)
    # model.restore()
    # model.evaluate(data)

    gt_pub = rospy.Publisher('gt_voxel_grid', OccupancyStamped, queue_size=10)
    completion_pub = rospy.Publisher('predicted_voxel_grid', OccupancyStamped, queue_size=10)

    for elem, _ in data:
        # IPython.embed()

        dim = elem.shape[0]
        
        gt_msg = conversions.vox_to_occupancy_stamped(elem.numpy(), dim=dim, scale=0.01, frame_id="base_frame")
        gt_pub.publish(gt_msg)

        elem = np.expand_dims(elem.numpy(), axis=0)
        # inference = model.model.predict(elem)
        # completion_msg = conversions.vox_to_occupancy_stamped(inference,
        #                                                       dim=dim, scale=0.01, frame_id="base_frame")
        # completion_pub.publish(completion_msg)
        

        
        rospy.sleep(1)

    

if __name__=="__main__":
    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")


    publish()
    # data_tools.write_shapenet_to_tfrecord()
    # data_tools.load_shapenet()
