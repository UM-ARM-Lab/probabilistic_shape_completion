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


    # test_dataset = data_tools._load_metadata_train_or_test(shapes="all", shuffle=False, prefix="test")
    ds, _ = data_tools.load_dataset("ycb", shuffle=False, metadata_only=True)
    # sharded_test_dataset = ds.shard(TOTAL_SHARDS, args.shard)
    sub_ds = ds.take(2)

    fits = plausiblility.compute_partial_icp_fit_dict(sub_ds, ds)
    plausiblility.save_plausibilities(fits, dataset_name="ycb")
    #
    # loaded_fits = plausiblility.load_plausibilities()
    print("Finished computing plausibilities")
