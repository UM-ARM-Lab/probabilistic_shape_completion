#! /usr/bin/env python
from shape_completion_training.model import plausiblility
from shape_completion_training.utils import data_tools
import argparse

TOTAL_SHARDS = 8

"""
This script computes plausibilies for a shard of the test dataset.
This can be manually run in multiple terminal windows to make use of multiple threads or multiple computers.

I would use multiprocessing, but that does not behave well with tensorflow
"""


def compute_plausibles_for_shard(shard):
    params = {'apply_slit_occlusion': False}

    train_ds, test_ds = data_tools.load_dataset("shapenet", shuffle=False, metadata_only=True)
    sharded_test_ds = test_ds.shard(TOTAL_SHARDS, shard)

    ref_size = 0
    for _ in sharded_test_ds:
        ref_size += 1

    # plausible_ds = test_ds.concatenate(train_ds.take(1*72))
    plausible_ds = test_ds.concatenate(train_ds.take(70 * 72))

    plausible_ds = data_tools.load_voxelgrids(plausible_ds)
    sharded_test_ds = data_tools.load_voxelgrids(sharded_test_ds)

    plausible_ds = data_tools.preprocess_test_dataset(plausible_ds, params)
    sharded_test_ds = data_tools.preprocess_test_dataset(sharded_test_ds, params)

    print("Computing shapenet plausibilities for shard {}/{}".format(shard, TOTAL_SHARDS))

    fits = plausiblility.compute_partial_icp_fit_dict(sharded_test_ds, plausible_ds, reference_ds_size=ref_size)
    plausiblility.save_plausibilities(fits, dataset_name="shapenet", identifier="_{}_{}".format(shard, TOTAL_SHARDS))
    #
    # loaded_fits = plausiblility.load_plausibilities()
    print("Finished computing plausibilities for shard {}/{}".format(shard, TOTAL_SHARDS))


def combine_shards():
    plausibles = {}
    for shard in range(TOTAL_SHARDS):
        identifier = "_{}_{}".format(shard, TOTAL_SHARDS)
        plausibles_shard = plausiblility.load_plausibilities(dataset_name="shapenet", identifier=identifier)
        for k, v in plausibles_shard:
            if k in plausibles:
                raise Exception("key {} already in another shard".format(k))
            plausibles[k] = v


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for sharded plausibility computation")
    parser.add_argument('shard', type=int)
    parser.add_argument('--combine_shards',
                        help='if passed this option, no plausibles will be '
                             'computed and all processed shards will be combined',
                        action='store_true')

    args = parser.parse_args()
    compute_plausibles_for_shard(args.shard)
