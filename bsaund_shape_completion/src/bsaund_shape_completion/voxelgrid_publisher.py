import rospy
from mps_shape_completion_visualization import conversions
from mps_shape_completion_msgs.msg import OccupancyStamped
from visualization_msgs.msg import Marker, MarkerArray
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
        pub_names = ["gt", "known_occ", "known_free", "predicted_occ", "predicted_free", "sampled_occ",
                     "conditioned_occ", "mismatch", "aux", "plausible"]
        for name in pub_names:
            self.add(name, name + "_voxel_grid")
        self.bb_pub = rospy.Publisher("bounding_box", MarkerArray, queue_size=1)

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

        if 'bounding_box' in elem:
            self.publish_bounding_box(elem["bounding_box"])

        def make_numpy(tensor_or_np):
            if tf.is_tensor(tensor_or_np):
                return tensor_or_np.numpy()
            return tensor_or_np

        print("Category: {}, id: {}, aug: {}".format(make_numpy(elem['category']),
                                                     make_numpy(elem['id']),
                                                     make_numpy(elem['augmentation'])))

    def publish_bounding_box(self, corners):
        if tf.is_tensor(corners):
            corners = corners.numpy()
        corner_markers = MarkerArray()

        for i, pt in enumerate(corners):
            marker = Marker()
            marker.id = i
            marker.header.frame_id = "object"
            marker.type = marker.SPHERE
            marker.action = marker.ADD
            marker.scale.x = 0.01
            marker.scale.y = 0.01
            marker.scale.z = 0.01
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.pose.orientation.w = 1.0
            marker.pose.position.x = pt[0]
            marker.pose.position.y = pt[1]
            marker.pose.position.z = pt[2]
            corner_markers.markers.append(marker)

            if i == 0:
                marker.color.g = 1.0
                marker.color.r = 1.0
            if i == 1:
                marker.color.b = 1.0
                marker.color.r = 1.0

        self.bb_pub.publish(corner_markers)

    def publish_inference(self, inference):
        self.publish("predicted_occ", inference["predicted_occ"])
        self.publish("predicted_free", inference["predicted_free"])
        if 'aux_occ' in inference:
            self.publish("aux_occ", inference("aux_occ"))
