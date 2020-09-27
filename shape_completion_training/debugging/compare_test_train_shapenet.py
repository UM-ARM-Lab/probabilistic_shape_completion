#!/usr/bin/env python
import argparse
import random

import rospy
import numpy as np

from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.model import model_evaluator, default_params
from shape_completion_training.utils import data_tools
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.model.other_model_architectures import sampling_tools
from shape_completion_training.voxelgrid import fit
from shape_completion_training.model import plausiblility, utils
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.voxelgrid.metrics import chamfer_distance
import tensorflow as tf
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher
from shape_completion_visualization.shape_selection import send_display_names_from_metadata
import pickle

"""
This function is based on a reviewer comment
We load the shapenet mugs and compare the predictions to the closest shape from the training dataset
This starts to examine the questions "Do shape completion networks just look up the nearest shape"
"""

parser = argparse.ArgumentParser(description='Publish shape data to RViz for viewing')
parser.add_argument('--trial')
parser.add_argument('--compute', action='store_true')
parser.add_argument('--analyze', action='store_true')

ARGS = parser.parse_args()


def compute():
    model_runner = ModelRunner(training=False, trial_path=ARGS.trial)

    default_dataset_params = default_params.get_default_params()
    dataset_params = default_dataset_params

    default_translations = {
        'translation_pixel_range_x': 0,
        'translation_pixel_range_y': 0,
        'translation_pixel_range_z': 0,
    }

    train_records, test_records = data_tools.load_dataset(dataset_name=dataset_params['dataset'],
                                                          metadata_only=True, shuffle=False)

    test_size = 0
    for _ in test_records:
        test_size += 1

    results = {}

    test_ds = data_tools.load_voxelgrids(test_records)
    test_ds = data_tools.preprocess_test_dataset(test_ds, dataset_params)
    for i, elem in test_ds.enumerate():
        print("Evaluating {}/{}".format(i.numpy(), test_size))
        # Computes and publishes the closest element in the training set to the test shape
        train_in_correct_augmentation = train_records.filter(lambda x: x['augmentation'] == elem['augmentation'])
        train_in_correct_augmentation = data_tools.load_voxelgrids(train_in_correct_augmentation)
        min_cd = np.inf
        closest_train = None
        for train_elem in train_in_correct_augmentation:
            # VG_PUB.publish("plausible", train_elem['gt_occ'])
            cd = chamfer_distance(elem['gt_occ'], train_elem['gt_occ'],
                                  scale=0.01, downsample=4)
            if cd < min_cd:
                min_cd = cd
                closest_train = train_elem['gt_occ']
                closest_train_id = train_elem['id']


        cd_test = []
        cd_train = []
        for _ in range(10):
            inference = model_runner.model(utils.add_batch_to_dict(elem))
            cd_test.append(chamfer_distance(elem['gt_occ'], inference['predicted_occ'],
                                            scale=0.01, downsample=4).numpy())
            cd_train.append(chamfer_distance(closest_train, inference['predicted_occ'],
                                             scale=0.01, downsample=4).numpy())

        results[(elem['id'].numpy().decode(), elem['augmentation'].numpy().decode())] = \
            {
            'closest_train': closest_train_id.numpy().decode(),
            'chamfer_dist_test': cd_test,
            'chamfer_dist_train': cd_train,
            }
        # if i.numpy() > 3:
        #     break
    return results


if ARGS.compute:
    results = compute()
    with open('../results/train_similarity/train_similarity.pkl', 'wb') as f:
        pickle.dump(results, f)

if ARGS.analyze:
    with open('../results/train_similarity/train_similarity.pkl', 'rb') as f:
        results = pickle.load(f)

    total_cd_test = []
    total_cd_train = []

    closer_train_count = 0
    total_count = 0

    for k, v in results.items():
        total_cd_test.append(np.mean(v['chamfer_dist_test']))
        total_cd_train.append(np.mean(v['chamfer_dist_train']))

        closer_train_count += np.sum(np.array(v['chamfer_dist_test']) > np.array(v['chamfer_dist_train']))
        total_count += len(v['chamfer_dist_test'])

    print("Average  est  dist is: {}".format(np.mean(total_cd_test)))
    print("Average train dist is: {}".format(np.mean(total_cd_train)))

    print("In total {}/{} samples were closer to the train".format(closer_train_count,
                                                                   total_count))

    results
