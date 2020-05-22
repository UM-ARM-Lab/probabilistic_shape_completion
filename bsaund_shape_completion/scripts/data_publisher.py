#!/usr/bin/env python
from __future__ import print_function

import argparse

import rospy
from mps_shape_completion_msgs.msg import OccupancyStamped
from mps_shape_completion_visualization import conversions


import numpy as np

import sys
import os

# sc_path = os.path.join(os.path.dirname(__file__), "../../")
# sys.path.append(sc_path)
from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.model import data_tools
from shape_completion_training.model import obj_tools
from shape_completion_training.model import nn_tools
from shape_completion_training.voxelgrid import metrics
from shape_completion_training import binvox_rw
# from bsaund_shape_completion import sampling_tools
from shape_completion_training.model import sampling_tools
from rviz_text_selection_panel_msgs.msg import TextSelectionOptions
from std_msgs.msg import String
import threading
import tensorflow as tf

ARGS = None
VG_PUB = None

options_pub = None
selected_sub = None

model = None

selection_map = {}
stop_current_sampler = None
sampling_thread = None


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
        if not elem.has_key('sampled_occ'):
            elem['sampled_occ'] = np.zeros(elem['gt_occ'].shape)
        self.publish("sampled_occ", elem["sampled_occ"])

        if not elem.has_key('conditioned_occ'):
            elem['conditioned_occ'] = np.zeros(elem['gt_occ'].shape)
        self.publish("conditioned_occ", elem["conditioned_occ"])

        def make_numpy(tensor_or_np):
            if tf.is_tensor(tensor_or_np):
                return tensor_or_np.numpy()
            return tensor_or_np

        print("Category: {}, id: {}, aug: {}".format(make_numpy(elem['shape_category']),
                                                     make_numpy(elem['id']),
                                                     make_numpy(elem['augmentation'])))


def to_msg(voxelgrid):
    return conversions.vox_to_occupancy_stamped(voxelgrid,
                                                dim = voxelgrid.shape[1],
                                                scale=0.01,
                                                frame_id = "object")


def publish_options(metadata):
    tso = TextSelectionOptions()

    for i, elem in metadata.enumerate():
        s = elem['id'].numpy() + elem['augmentation'].numpy()
        selection_map[s] = i
        tso.options.append(s)

    options_pub.publish(tso)


def publish_inference(inference):
    VG_PUB.publish("predicted_occ", inference["predicted_occ"])
    VG_PUB.publish("predicted_free", inference["predicted_free"])
    if inference.has_key('aux_occ'):
        VG_PUB.publish("aux_occ", inference("aux_occ"))


def run_inference(elem):
    if not ARGS.use_best_iou:
        return model.model(elem)

    best_iou = 0.0
    best_inference = None
    for _ in range(300):
        inference = model.model(elem)
        iou = metrics.iou(elem['gt_occ'], inference['predicted_occ'])
        if ARGS.publish_each_sample:
            publish_inference(inference)
        if iou > best_iou:
            best_iou = iou
            best_inference = inference
    return best_inference


def publish_selection(metadata, str_msg):
    translation = 0
    
    ds = metadata.skip(selection_map[str_msg.data]).take(1)
    ds = data_tools.load_voxelgrids(ds)
    ds = data_tools.simulate_input(ds, 0, 0, 0)
    # sim_input_fn = lambda gt: data_tools.simulate_first_n_input(gt, 64**3 * 4/8)
    # sim_input_fn = lambda gt: data_tools.simulate_first_n_input(gt, 64**3)
    
    # ds = data_tools.simulate_input(ds, translation, translation, translation,
    #                                sim_input_fn=sim_input_fn)
    # ds = data_tools.simulate_condition_occ(ds, turn_on_prob = 0.00001, turn_off_prob=0.1)
    # ds = data_tools.simulate_condition_occ(ds, turn_on_prob = 0.00000, turn_off_prob=0.0)
    
    # ds = data_tools.simulate_partial_completion(ds)
    # ds = data_tools.simulate_random_partial_completion(ds)

    elem_raw = next(ds.__iter__())
    elem = {}

    for k in elem_raw.keys():
        elem_raw[k] = tf.expand_dims(elem_raw[k], axis=0)
    
    for k in elem_raw.keys():
        elem[k] = elem_raw[k].numpy()
    VG_PUB.publish_elem(elem)

    if model is None:
        return
        
    elem = sampling_tools.prepare_for_sampling(elem)

    inference = run_inference(elem)
    publish_inference(inference)

    mismatch = np.abs(elem['gt_occ'] - inference['predicted_occ'].numpy())
    VG_PUB.publish("mismatch", mismatch)
    # mismatch_pub.publish(to_msg(mismatch))
    print("There are {} mismatches".format(np.sum(mismatch > 0.5)))

    def multistep_error(elem, inference):
        a = inference['predicted_occ']
        # a = inference['predicted_occ'] +  elem['known_occ'] - elem['known_free']
        elem['conditioned_occ'] = np.float32(a > 0.5)
        inference = model.model(elem)
        mismatch = np.abs(elem['gt_occ'] - inference['predicted_occ'].numpy())
        VG_PUB.publish("mismatch", mismatch)
        # mismatch_pub.publish(to_msg(mismatch))
        return elem, inference
        
    if ARGS.multistep:
        for _ in range(5):
            rospy.sleep(1)
            elem, inference = multistep_error(elem, inference)

    metric = metrics.p_correct_geometric_mean(inference['predicted_occ'], elem['gt_occ'])
    print("p_correct_geometric_mean: {}".format(metric.numpy()))
    print("p_correct: {}".format(metrics.p_correct(inference['predicted_occ'], elem['gt_occ'])))
    print("iou: {}".format(metrics.iou(elem['gt_occ'], inference['predicted_occ'])))

    if ARGS.sample:
        global stop_current_sampler
        global sampling_thread
        
        # print("Stopping old worker")
        stop_current_sampler = True
        if sampling_thread is not None:
            sampling_thread.join()
        
        sampling_thread = threading.Thread(target=sampler_worker, args=(elem,))
        sampling_thread.start()


def sampler_worker(elem):
    global stop_current_sampler
    stop_current_sampler = False

    print()
    for i in range(200):
        if stop_current_sampler:
            return
        rospy.sleep(0.01)
    
    # sampler = sampling_tools.UnknownSpaceSampler(elem)
    sampler = sampling_tools.EfficientCNNSampler(elem)
    # sampler = sampling_tools.MostConfidentSampler(elem)
    inference = model.model(elem)

    finished = False
    prev_ct = 0

    while not finished and not stop_current_sampler:
        try:
            elem, inference = sampler.sample(model, elem, inference)
        except StopIteration:
            finished = True

        if sampler.ct - prev_ct >= 100 or finished:
            prev_ct = sampler.ct
            publish_np_elem(elem)
            publish_inference(inference)
    print("Sampling complete")
    # IPython.embed()


def publish_object_transform_old():
    """
    This is deprecated and will be removed
    1) Use `roslaunch bsaund_shape_completion shape_completion.launch` and this is not necessary
    2) Use mps_shape_completion_visualization/quick_publish.py/publish_object_transform
    """


def load_network():
    global model
    if ARGS.trial is None:
        print("Not loading any inference model")
        return
    model = ModelRunner(training=False, trial_path=ARGS.trial)


def parser():
    global ARGS
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--sample', help='foo help', action='store_true')
    parser.add_argument('--use_best_iou', help='foo help', action='store_true')
    parser.add_argument('--publish_each_sample', help='foo help', action='store_true')
    parser.add_argument('--multistep', action='store_true')
    parser.add_argument('--trial')

    ARGS = parser.parse_args()


if __name__=="__main__":
    parser()
    
    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    records = data_tools.load_shapenet_metadata(shuffle=False)
    load_network()

    pub_names = ["gt", "known_occ", "known_free", "predicted_occ", "predicted_free", "sampled_occ",
                 "conditioned_occ", "mismatch", "aux"]
    VG_PUB = VoxelgridPublisher()
    for name in pub_names:
        VG_PUB.add(name, name+"_voxel_grid")

    options_pub = rospy.Publisher('shapenet_options', TextSelectionOptions, queue_size=1)
    selected_sub = rospy.Subscriber('/shapenet_selection', String,
                                    lambda x: publish_selection(records, x))

    publish_options(records)

    rospy.spin()

    









