#!/usr/bin/env python
from __future__ import print_function

import argparse
import random

import rospy
import numpy as np

from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.model import model_evaluator, default_params
from shape_completion_training.utils import data_tools
from shape_completion_training.model.other_model_architectures import sampling_tools
from shape_completion_training.model import plausiblility
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.voxelgrid.metrics import chamfer_distance
import tensorflow as tf
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher
from shape_completion_visualization.shape_selection import send_display_names_from_metadata

ARGS = None
VG_PUB = None

model_runner = None
dataset_params = None
sampling_thread = None

default_dataset_params = default_params.get_default_params()

default_translations = {
    'translation_pixel_range_x': 0,
    'translation_pixel_range_y': 0,
    'translation_pixel_range_z': 0,
}


def run_inference(elem):
    if ARGS.publish_each_sample:
        for particle in model_evaluator.sample_particles(model_runner.model, elem, 100):
            VG_PUB.publish("predicted_occ", particle)
            rospy.sleep(0.1)

    inference = model_runner.model(elem)
    if not ARGS.publish_nearest_sample:
        VG_PUB.publish_inference(inference)

    min_cd = np.inf
    best_fit = None
    if ARGS.publish_nearest_plausible:
        for plausible in plausiblility.get_plausibilities_for(data_tools.get_unique_name(elem)[0],
                                                              model_runner.params['dataset']):
            elem_name, T, p, oob = plausible
            sn = data_tools.get_addressible_dataset(dataset_name=model_runner.params['dataset'])
            plausible_elem = sn.get(elem_name)
            fitted = conversions.transform_voxelgrid(plausible_elem['gt_occ'], T, scale=0.01)
            VG_PUB.publish("plausible", fitted)
            cd = chamfer_distance(tf.cast(inference['predicted_occ'] > 0.5, tf.float32), fitted,
                                  scale=0.01, downsample=4)
            print("Chamfer distance: {}".format(cd))
            if cd < min_cd:
                min_cd = cd
                best_fit = fitted

        VG_PUB.publish("plausible", best_fit)
        print("Best Fit CD: {}".format(min_cd))

    if ARGS.publish_nearest_sample:
        min_cd = np.inf
        best_fit = None
        plausibles = plausiblility.get_plausibilities_for(data_tools.get_unique_name(elem)[0],
                                                          model_runner.params['dataset'])
        plausible = random.choice(plausibles)
        elem_name, T, p, oob = plausible
        sn = data_tools.get_addressible_dataset(dataset_name=model_runner.params['dataset'])
        plausible_elem = sn.get(elem_name)
        fitted = conversions.transform_voxelgrid(plausible_elem['gt_occ'], T, scale=0.01)
        VG_PUB.publish("plausible", fitted)
        rospy.sleep(0.5)

        for particle in model_evaluator.sample_particles(model_runner.model, elem, 20):
            VG_PUB.publish("predicted_occ", particle)
            rospy.sleep(0.1)
            cd = chamfer_distance(tf.cast(particle > 0.5, tf.float32), fitted,
                                  scale=0.01, downsample=4)
            print("Chamfer distance: {}".format(cd))
            if cd < min_cd:
                min_cd = cd
                best_fit = particle
        VG_PUB.publish("predicted_occ", best_fit)
        print("Best Fit CD: {}".format(min_cd))
    return inference


def publish_selection(metadata, ind, str_msg):
    if ind == 0:
        print("Skipping first display")
        return

    # translation = 0

    ds = metadata.skip(ind).take(1)
    ds = data_tools.load_voxelgrids(ds)
    ds = data_tools.preprocess_test_dataset(ds, dataset_params)

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

    inference = run_inference(elem_raw)

    mismatch = np.abs(elem['gt_occ'] - inference['predicted_occ'].numpy())
    VG_PUB.publish("mismatch", mismatch)
    # mismatch_pub.publish(to_msg(mismatch))
    print("There are {} mismatches".format(np.sum(mismatch > 0.5)))


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
    parser.add_argument('--publish_each_sample', help='foo help', action='store_true')
    parser.add_argument('--publish_nearest_plausible', help='foo help', action='store_true')
    parser.add_argument('--publish_nearest_sample', help='foo help', action='store_true')
    parser.add_argument('--trial')

    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_command_line_args()

    rospy.init_node('shape_publisher')
    rospy.loginfo("Data Publisher")

    load_network()

    dataset_params = default_dataset_params
    if model_runner is not None:
        dataset_params.update(model_runner.params)
        dataset_params.update({
            "slit_start": 32,
            "slit_width": 32,
        })
    # dataset_params.update({
    #     "apply_depth_sensor_noise": True,
    # })

    dataset_params.update(default_translations)
    train_records, test_records = data_tools.load_dataset(dataset_name=dataset_params['dataset'],
                                                          metadata_only=True, shuffle=False)

    VG_PUB = VoxelgridPublisher()

    # selection_sub = send_display_names_from_metadata(train_records, publish_selection)
    selection_sub = send_display_names_from_metadata(test_records, publish_selection)

    rospy.spin()
