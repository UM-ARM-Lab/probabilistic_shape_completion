#! /usr/bin/env python

from shape_completion_training.model import data_tools
from shape_completion_training.model.modelrunner import ModelRunner
from bsaund_shape_completion import voxelgrid_publisher
import tensorflow as tf
import rospy
from shape_completion_training.model.utils import add_batch_to_dict
from shape_completion_training.voxelgrid.bounding_box import unflatten_bounding_box, flatten_bounding_box


def get_flow():
    mr = ModelRunner(training=False, trial_path="Flow/June_13_13-09-11_4bef25fbe3")
    return mr.model.flow

def view_flow():
    flow = get_flow()
    vg_pub = voxelgrid_publisher.VoxelgridPublisher()
    X = flow.distribution.sample(10)
    Y = flow.bijector.forward(X)
    for bb_flat in Y:
        bb = tf.reshape(bb_flat, (8, 3))
        print(bb.numpy())
        vg_pub.publish_bounding_box(bb)
        rospy.sleep(1)

    print("done")


def view_inferred_bounding_box():
    vg_pub = voxelgrid_publisher.VoxelgridPublisher()
    sn = data_tools.get_shapenet()
    mr = ModelRunner(training=False, trial_path="Normalizing_AE/June_17_20-47-16_bdb4e72791")
    flow = get_flow()
    for i in range(100):
        elem = sn.get(sn.train_names[i])
        elem = add_batch_to_dict(elem)
        output = mr.model(elem)
        flat_bb = flow.bijector.forward(output['mean'])
        flat_bb_1 = mr.model.flow.bijector.forward(output['mean'])
        print(flat_bb - flat_bb_1)
        bb = unflatten_bounding_box(flat_bb)
        vg_pub.publish_elem(elem)
        rospy.sleep(1)

        l1 = mr.model.flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        bb = unflatten_bounding_box(flow.bijector.forward(l1))


        vg_pub.publish_bounding_box(bb)
        rospy.sleep(1)

if __name__ == "__main__":
    rospy.init_node("bounding_box_flow_publisher")
    # view_flow()
    view_inferred_bounding_box()

    # mr.train_and_test(sn.train_ds)
