#! /usr/bin/env python
from shape_completion_training.model import plausiblility
from shape_completion_training.utils import data_tools
import argparse


TOTAL_SHARDS = 20

"""
This script computes plausibilies for a shard of the test dataset.
This can be manually run in multiple terminal windows to make use of multiple threads or multiple computers.

I would use multiprocessing, but that does not behave well with tensorflow
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for sharded plausibility computation")
    parser.add_argument('shard', type=int)

    args = parser.parse_args()


    test_dataset = data_tools._load_metadata_train_or_test(shapes="all", shuffle=False, prefix="test")
    sharded_test_dataset = test_dataset.shard(TOTAL_SHARDS, args.shard)

    fits = plausiblility.compute_partial_icp_fit_dict(sharded_test_dataset, test_dataset)
    plausiblility.save_plausibilities(fits, identifier="_{}_{}".format(args.shard, TOTAL_SHARDS))
    #
    # loaded_fits = plausiblility.load_plausibilities()
    print("Finished computing plausibilities for shard {}/{}".format(args.shard, TOTAL_SHARDS))
