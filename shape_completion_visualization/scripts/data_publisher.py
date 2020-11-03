#!/usr/bin/env python
from __future__ import print_function

import argparse
import random

import rospy
import numpy as np

from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.model import default_params
from shape_completion_training.utils import data_tools
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.model.other_model_architectures import sampling_tools
from shape_completion_training.voxelgrid import fit
from shape_completion_training.plausible_diversity import model_evaluator, plausiblility
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.voxelgrid.metrics import chamfer_distance
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher
from shape_completion_visualization.shape_selection import send_display_names_from_metadata

import threading
import tensorflow as tf

ARGS = None
VG_PUB = None

model_runner = None
dataset_params = None

stop_current_sampler = None
sampling_thread = None

default_dataset_params = default_params.get_default_params()

default_translations = {
    'translation_pixel_range_x': 0,
    'translation_pixel_range_y': 0,
    'translation_pixel_range_z': 0,
}

def wip_enforce_contact(elem):
    inference = model_runner.model(elem)
    VG_PUB.publish_inference(inference)
    pssnet = model_runner.model
    latent = tf.Variable(pssnet.sample_latent(elem))
    VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
    known_contact = tf.Variable(tf.zeros((1, 64, 64, 64, 1)))
    known_contact = known_contact[0, 50, 32, 32, 0].assign(1)
    VG_PUB.publish('aux', known_contact)

    rospy.sleep(2)

    for i in range(100):
        pssnet.grad_step_towards_output(latent, known_contact)
        VG_PUB.publish('predicted_occ', pssnet.decode(latent, apply_sigmoid=True))
    return pssnet.decode(latent, apply_sigmoid=True)


def run_inference(elem):
    if ARGS.enforce_contact:
        return wip_enforce_contact(elem)


    if ARGS.publish_closest_train:
        # Computes and publishes the closest element in the training set to the test shape
        train_in_correct_augmentation = train_records.filter(lambda x: x['augmentation'] == elem['augmentation'][0])
        train_in_correct_augmentation = data_tools.load_voxelgrids(train_in_correct_augmentation)
        min_cd = np.inf
        closest_train = None
        for train_elem in train_in_correct_augmentation:
            VG_PUB.publish("plausible", train_elem['gt_occ'])
            cd = chamfer_distance(elem['gt_occ'], train_elem['gt_occ'],
                                  scale=0.01, downsample=4)
            if cd < min_cd:
                min_cd = cd
                closest_train = train_elem['gt_occ']
            VG_PUB.publish("plausible", closest_train)


    if ARGS.publish_each_sample:
        for particle in model_evaluator.sample_particles(model_runner.model, elem, 20):
            VG_PUB.publish("predicted_occ", particle)
            rospy.sleep(0.1)

    # raw_input("Ready to display best?")

    inference = model_runner.model(elem)
    if not ARGS.publish_nearest_sample:
        VG_PUB.publish_inference(inference)

    min_cd = np.inf
    best_fit = None
    if ARGS.publish_nearest_plausible:
        for plausible in plausiblility.get_plausibilities_for(data_tools.get_unique_name(elem, has_batch_dim=True),
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
        plausibles = plausiblility.get_plausibilities_for(data_tools.get_unique_name(elem, has_batch_dim=True),
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
        # VG_PUB.publish("plausible", best_fit)
        print("Best Fit CD: {}".format(min_cd))

        # for  in valid_fits:
        #
        #     VG_PUB.publish("plausible", fitted)
        #     p = shape_completion_training.model.observation_model.observation_likelihood_geometric_mean(
        #         reference['gt_occ'],
        #         fitted,
        #         std_dev_in_voxels=2)
        #     print("Best fit for {}: p={}".format(data_tools.get_unique_name(elem), p))
        #     rospy.sleep(0.1)

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
    ds = data_tools.preprocess_test_dataset(ds, dataset_params)
    # ds = data_tools.apply_sensor_noise(ds)
    # ds = data_tools.simulate_input(ds, 0, 0, 0)
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

    inference = run_inference(elem_raw)
    if ARGS.enforce_contact:
        return

    # VG_PUB.publish_inference(inference)

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
        for _ in range(20):
            # rospy.sleep(1)
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
    parser.add_argument('--publish_nearest_plausible', help='foo help', action='store_true')
    parser.add_argument('--publish_nearest_sample', help='foo help', action='store_true')
    parser.add_argument('--multistep', action='store_true')
    parser.add_argument('--trial')
    parser.add_argument('--publish_closest_train', action='store_true')
    parser.add_argument('--enforce_contact', action='store_true')

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
