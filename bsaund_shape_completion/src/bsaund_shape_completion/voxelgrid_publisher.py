import rospy
from mps_shape_completion_visualization import conversions
from mps_shape_completion_msgs.msg import OccupancyStamped
import tensorflow as tf
import numpy as np


def to_msg(voxelgrid):
    return conversions.vox_to_occupancy_stamped(voxelgrid,
                                                dim=voxelgrid.shape[1],
                                                scale=0.01,
                                                frame_id="object")


class VoxelgridPublisher:
    def __init__(self):
        self.pubs = {}

    def add(self, short_name, topic):
        self.pubs[short_name] = rospy.Publisher(topic, OccupancyStamped, queue_size=1)

    def publish(self, short_name, voxelgrid):
        if tf.is_tensor(voxelgrid):
            voxelgrid = voxelgrid.numpy()
        self.pubs[short_name].publish(to_msg(voxelgrid))

    def publish_elem(self, elem):
        self.publish("gt", elem["gt_occ"])
        self.publish("known_occ", elem["known_occ"])
        self.publish("known_free", elem["known_free"])
        if 'sampled_occ' not in elem:
            elem['sampled_occ'] = np.zeros(elem['gt_occ'].shape)
        self.publish("sampled_occ", elem["sampled_occ"])

        if 'conditioned_occ' not in elem:
            elem['conditioned_occ'] = np.zeros(elem['gt_occ'].shape)
        self.publish("conditioned_occ", elem["conditioned_occ"])

        def make_numpy(tensor_or_np):
            if tf.is_tensor(tensor_or_np):
                return tensor_or_np.numpy()
            return tensor_or_np

        print("Category: {}, id: {}, aug: {}".format(make_numpy(elem['shape_category']),
                                                     make_numpy(elem['id']),
                                                     make_numpy(elem['augmentation'])))

    def publish_inference(self, inference):
        self.publish("predicted_occ", inference["predicted_occ"])
        self.publish("predicted_free", inference["predicted_free"])
        if 'aux_occ' in inference:
            self.publish("aux_occ", inference("aux_occ"))
