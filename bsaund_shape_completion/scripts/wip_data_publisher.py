#!/usr/bin/env python
from __future__ import print_function

import argparse
import rospy
import numpy as np

from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.utils import data_tools
from shape_completion_training.model.utils import add_batch_to_dict, numpyify
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.model import sampling_tools
import tensorflow as tf
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher
from bsaund_shape_completion.shape_selection import send_display_names_from_metadata

"""
Simplified version of data_publisher that I will make changes to, to explore specific models
"""

ARGS = None
VG_PUB = None

model_runner = None
# model_evaluator = None

stop_current_sampler = None
sampling_thread = None


def display_augmented_vae(elem):
    inference = model_runner.model(elem)
    VG_PUB.publish_inference(inference)
    mean_f, mean_angle = model_runner.model.split_angle(inference['latent_mean'])
    logvar_f, logvar_angle = model_runner.model.split_angle(inference['latent_logvar'])
    sampled_f, sampled_angle = model_runner.model.split_angle(inference['sampled_latent'])

    print("{} +/- {} angle".format(mean_angle.numpy()[0, 0], tf.sqrt(tf.exp(logvar_angle)).numpy()[0, 0]))
    print("{} sampled angle".format(sampled_angle.numpy()[0, 0]))
    print("{} actual angle".format(elem['angle'][0]))

    return inference


def run_inference(elem):
    special_displays = {'Augmented_VAE': display_augmented_vae}

    try:
        return special_displays[model_runner.params['network']](elem)
    except KeyError as e:
        pass

    inference = model_runner.model(elem)
    VG_PUB.publish_inference(inference)
    return inference


def publish_selection(metadata, ind, str_msg):
    if ind == 0:
        print("Skipping first display")
        return

    ds = metadata.skip(ind).take(1)
    ds = data_tools.load_voxelgrids(ds)
    ds = data_tools.simulate_input(ds, 0, 0, 0)
    ds = data_tools.apply_slit_occlusion(ds)

    # elem = numpyify(next(ds.__iter__()))
    # data_tools.simulate_2_5D_with_occlusion(elem['gt_occ'], 1, 32)

    elem_raw = next(ds.__iter__())
    elem_raw = add_batch_to_dict(elem_raw)
    elem = numpyify(elem_raw)
    VG_PUB.publish_elem(elem)

    if model_runner is None:
        return

    elem = sampling_tools.prepare_for_sampling(elem)

    inference = run_inference(elem)

    mismatch = np.abs(elem['gt_occ'] - inference['predicted_occ'].numpy())
    VG_PUB.publish("mismatch", mismatch)
    # mismatch_pub.publish(to_msg(mismatch))
    print("There are {} mismatches".format(np.sum(mismatch > 0.5)))

    metric = metrics.p_correct_geometric_mean(inference['predicted_occ'], elem['gt_occ'])
    print("p_correct_geometric_mean: {}".format(metric.numpy()))
    print("p_correct: {}".format(metrics.p_correct(inference['predicted_occ'], elem['gt_occ'])))
    print("iou: {}".format(metrics.iou(elem['gt_occ'], inference['predicted_occ'])))


def load_network():
    global model_runner
    if ARGS.trial is None:
        print("Not loading any inference model")
        return
    model_runner = ModelRunner(training=False, trial_path=ARGS.trial)
    # model_evaluator = ModelEvaluator(model_runner.model)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
    parser.add_argument('--trial')

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_command_line_args()

    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    # train_records, test_records = data_tools.load_shapenet_metadata(shuffle=False)
    train_records, test_records = data_tools.load_ycb_metadata(shuffle=False)
    load_network()

    VG_PUB = VoxelgridPublisher()

    selection_sub = send_display_names_from_metadata(train_records, publish_selection)

    rospy.spin()
