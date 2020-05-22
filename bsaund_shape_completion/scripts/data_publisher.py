#!/usr/bin/env python
from __future__ import print_function

import argparse
import rospy
import numpy as np

from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.model.model_evaluator import ModelEvaluator
from shape_completion_training.model import data_tools
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.model import sampling_tools
from rviz_text_selection_panel_msgs.msg import TextSelectionOptions
from std_msgs.msg import String
import threading
import tensorflow as tf
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher

ARGS = None
VG_PUB = None

options_pub = None
selected_sub = None

model_runner = None
model_evaluator = None

selection_map = {}
stop_current_sampler = None
sampling_thread = None


def publish_options(metadata):
    tso = TextSelectionOptions()

    for i, elem in metadata.enumerate():
        s = elem['id'].numpy() + elem['augmentation'].numpy()
        selection_map[s] = i
        tso.options.append(s)

    options_pub.publish(tso)


def run_inference(elem):
    if not ARGS.use_best_iou:
        return model_runner.model(elem)

    # best_iou = 0.0
    # best_inference = None
    # for _ in range(300):
    #     inference = model_runner.model(elem)
    #     iou = metrics.iou(elem['gt_occ'], inference['predicted_occ'])
    #     if ARGS.publish_each_sample:
    #         VG_PUB.publish_inference(inference)
    #     if iou > best_iou:
    #         best_iou = iou
    #         best_inference = inference
    # if ARGS.publish_each_sample:
    #     raw_input("Ready to publish final sample?")
    sample_evaluation = model_evaluator.evaluate_element(elem, num_samples=10)
    if ARGS.publish_each_sample:
        for particle in sample_evaluation.particles:
            VG_PUB.publish("predicted_occ", particle)
            rospy.sleep(0.5)

    # raw_input("Ready to display best?")
    inference = model_evaluator.model(elem)
    inference["predicted_occ"] = sample_evaluation.get_best_particle(
        metric=lambda a, b: -metrics.chamfer_distance(a, b, scale=0.01, downsample=2).numpy())
    return inference


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

    if model_runner is None:
        return

    elem = sampling_tools.prepare_for_sampling(elem)

    inference = run_inference(elem)
    VG_PUB.publish_inference(inference)

    mismatch = np.abs(elem['gt_occ'] - inference['predicted_occ'].numpy())
    VG_PUB.publish("mismatch", mismatch)
    # mismatch_pub.publish(to_msg(mismatch))
    print("There are {} mismatches".format(np.sum(mismatch > 0.5)))

    def multistep_error(elem, inference):
        a = inference['predicted_occ']
        # a = inference['predicted_occ'] +  elem['known_occ'] - elem['known_free']
        elem['conditioned_occ'] = np.float32(a > 0.5)
        inference = model_runner.model(elem)
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
    inference = model_runner.model(elem)

    finished = False
    prev_ct = 0

    while not finished and not stop_current_sampler:
        try:
            elem, inference = sampler.sample(model_runner, elem, inference)
        except StopIteration:
            finished = True

        if sampler.ct - prev_ct >= 100 or finished:
            prev_ct = sampler.ct
            VG_PUB.publish_elem(elem)
            VG_PUB.publish_inference(inference)
    print("Sampling complete")
    # IPython.embed()


def publish_object_transform_old():
    """
    This is deprecated and will be removed
    1) Use `roslaunch bsaund_shape_completion shape_completion.launch` and this is not necessary
    2) Use mps_shape_completion_visualization/quick_publish.py/publish_object_transform
    """
    pass


def load_network():
    global model_runner
    global model_evaluator
    if ARGS.trial is None:
        print("Not loading any inference model")
        return
    model_runner = ModelRunner(training=False, trial_path=ARGS.trial)
    model_evaluator = ModelEvaluator(model_runner.model)


def parse_command_line_args():
    global ARGS
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--sample', help='foo help', action='store_true')
    parser.add_argument('--use_best_iou', help='foo help', action='store_true')
    parser.add_argument('--publish_each_sample', help='foo help', action='store_true')
    parser.add_argument('--multistep', action='store_true')
    parser.add_argument('--trial')

    ARGS = parser.parse_args()


if __name__ == "__main__":
    parse_command_line_args()

    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    records = data_tools.load_shapenet_metadata(shuffle=False)
    load_network()

    pub_names = ["gt", "known_occ", "known_free", "predicted_occ", "predicted_free", "sampled_occ",
                 "conditioned_occ", "mismatch", "aux"]
    VG_PUB = VoxelgridPublisher()
    for name in pub_names:
        VG_PUB.add(name, name + "_voxel_grid")

    options_pub = rospy.Publisher('shapenet_options', TextSelectionOptions, queue_size=1)
    selected_sub = rospy.Subscriber('/shapenet_selection', String,
                                    lambda x: publish_selection(records, x))

    publish_options(records)

    rospy.spin()
