#! /usr/bin/env python

from shape_completion_training.model import data_tools
from shape_completion_training.model.modelrunner import ModelRunner
from bsaund_shape_completion import voxelgrid_publisher
import tensorflow as tf
import rospy

if __name__ == "__main__":
    rospy.init_node("bounding_box_flow_publisher")
    sn = data_tools.get_shapenet()

    mr = ModelRunner(training=False, trial_path="Flow/June_13_13-09-11_4bef25fbe3")
    # IPython.embed()
    vg_pub = voxelgrid_publisher.VoxelgridPublisher()
    X = mr.model.flow.distribution.sample(10)
    Y = mr.model.flow.bijector.forward(X)
    for bb_flat in Y:
        bb = tf.reshape(bb_flat, (8,3))
        print(bb.numpy())
        vg_pub.publish_bounding_box(bb)
        rospy.sleep(1)

    print("done")
    # mr.train_and_test(sn.train_ds)
