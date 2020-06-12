#!/usr/bin/env python
from __future__ import print_function

import argparse
import rospy
import numpy as np

from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.model import model_evaluator
from shape_completion_training.model import data_tools
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.model import sampling_tools
from shape_completion_training.voxelgrid import fit
from shape_completion_training.voxelgrid import bounding_box
import threading
import tensorflow as tf
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher
from bsaund_shape_completion.shape_selection import send_display_names_from_metadata

ARGS = None
VG_PUB = None

model_runner = None
# model_evaluator = None

stop_current_sampler = None
sampling_thread = None


def run_inference(elem):
    if not ARGS.publish_each_sample and not ARGS.use_best_iou:
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
    # sample_evaluation = model_evaluator.evaluate_element(elem, num_samples=10)
    if ARGS.publish_each_sample:
        for particle in model_evaluator.sample_particles(model_runner.model, elem, 20):
            VG_PUB.publish("predicted_occ", particle)
            rospy.sleep(0.5)

    # raw_input("Ready to display best?")
    inference = model_runner.model(elem)
    # inference["predicted_occ"] = sample_evaluation.get_best_particle(
    #     metric=lambda a, b: -metrics.chamfer_distance(a, b, scale=0.01, downsample=2).numpy())
    # VG_PUB.publish_inference(inference)
    # fit_to_particles(train_records, sample_evaluation)

    return inference


def fit_to_particle(metadata, particle):

    ds = metadata.shuffle(10000).take(10)
    ds = data_tools.load_voxelgrids(ds)
    # ds = data_tools.simulate_input(ds, 0, 0, 0)
    possibles = [e['gt_occ'] for e in ds.__iter__()]
    fitted = [fit.icp(possible, particle, scale=0.01) for possible in possibles]
    for f in fitted:
        VG_PUB.publish("gt", f)
        rospy.sleep(1)


def fit_to_particles(metadata, sample_evaluation):
    if not ARGS.fit_to_particles:
        return
    for i, particle in enumerate(sample_evaluation.particles):
        raw_input("Ready to fit to particle {}".format(i))
        fit_to_particle(metadata, particle)


def publish_selection(metadata, ind, str_msg):
    if ind == 0:
        print("Skipping first display")
        return

    # translation = 0

    ds = metadata.skip(ind).take(1)
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
    # VG_PUB.publish_bounding_box(bounding_box.get_aabb(elem["gt_occ"]))
    # VG_PUB.publish("predicted_occ", bounding_box.get_bounding_box_for_elem(elem))
    # bounds = bounding_box.get_bounding_box_for_elem(elem)
    # VG_PUB.publish_bounding_box(bounds)

    if model_runner is None:
        return

    elem = sampling_tools.prepare_for_sampling(elem)

    inference = run_inference(elem)
    VG_PUB.publish_inference(inference)

    # fit_to_particles(metadata)

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


def load_network():
    global model_runner
    # global model_evaluator
    if ARGS.trial is None:
        print("Not loading any inference model")
        return
    model_runner = ModelRunner(training=False, trial_path=ARGS.trial)
    # model_evaluator = ModelEvaluator(model_runner.model)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--sample', help='foo help', action='store_true')
    parser.add_argument('--use_best_iou', help='foo help', action='store_true')
    parser.add_argument('--publish_each_sample', help='foo help', action='store_true')
    parser.add_argument('--fit_to_particles', help='foo help', action='store_true')
    parser.add_argument('--multistep', action='store_true')
    parser.add_argument('--trial')

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_command_line_args()

    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    train_records, test_records = data_tools.load_shapenet_metadata(shuffle=False)
    load_network()

    VG_PUB = VoxelgridPublisher()

    selection_sub = send_display_names_from_metadata(test_records, publish_selection)

    rospy.spin()
