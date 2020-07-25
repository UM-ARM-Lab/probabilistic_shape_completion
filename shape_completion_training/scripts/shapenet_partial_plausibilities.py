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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for sharded plausibility computation")
    parser.add_argument('shard', type=int)

    args = parser.parse_args()

    params = {'apply_slit_occlusion': False}

    train_ds, test_ds = data_tools.load_dataset("shapenet", shuffle=False, metadata_only=True)
    sharded_test_ds = test_ds.shard(TOTAL_SHARDS, args.shard)

    ref_size = 0
    for _ in sharded_test_ds:
        ref_size += 1

    plausible_ds = test_ds.concatenate(train_ds)

    plausible_ds = data_tools.load_voxelgrids(plausible_ds)
    sharded_test_ds = data_tools.load_voxelgrids(sharded_test_ds)

    plausible_ds = data_tools.preprocess_test_dataset(plausible_ds, params)
    sharded_test_ds = data_tools.preprocess_test_dataset(sharded_test_ds, params)

    fits = plausiblility.compute_partial_icp_fit_dict(sharded_test_ds, plausible_ds)
    plausiblility.save_plausibilities(fits, identifier="_{}_{}".format(args.shard, TOTAL_SHARDS))
    #
    # loaded_fits = plausiblility.load_plausibilities()
    print("Finished computing plausibilities for shard {}/{}".format(args.shard, TOTAL_SHARDS))
