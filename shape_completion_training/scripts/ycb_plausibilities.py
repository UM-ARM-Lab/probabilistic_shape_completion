#! /usr/bin/env python
from shape_completion_training.model import plausiblility
from shape_completion_training.utils import data_tools
import argparse

TOTAL_SHARDS = 8

"""
Plausibilities for the YCB dataset with slit occlusion
"""

# 6 slit params
# slit_start = 28
# slit_width = 6

# 30 slit params
slit_start = 17
slit_width = 30


def compute_plausibles_for_shard(shard):

    _, ds = data_tools.load_dataset("ycb", shuffle=False, metadata_only=True)


    params = {'apply_slit_occlusion': True,
              'slit_start': slit_start,
              'slit_width': slit_width}


    reference_ds = ds.shard(TOTAL_SHARDS, shard)
    ref_size = 0
    for _ in ds:
        ref_size += 1
    reference_ds = data_tools.load_voxelgrids(reference_ds)
    reference_ds = data_tools.preprocess_test_dataset(reference_ds, params)

    ds = data_tools.load_voxelgrids(ds)

    single_match_ds = data_tools.simulate_input(ds, 0, 0, 0, sim_input_fn=data_tools.simulate_2_5D_input)
    match_ds =  data_tools.apply_fixed_slit_occlusion(single_match_ds, 32, 6)

    for slit_min in range(32-10, 32+10, 2):
        if slit_min == 32:
            continue
        match_ds = match_ds.concatenate(data_tools.apply_fixed_slit_occlusion(single_match_ds, slit_min, slit_width))

    mask = data_tools.get_slit_occlusion_2D_mask(slit_start, slit_width, (64, 64))


    print("Computing shapenet plausibilities for shard {}/{}".format(shard, TOTAL_SHARDS))

    # reference_ds = reference_ds.take(10) # Comment this in to do a quick run to verify computation

    fits = plausiblility.compute_partial_icp_fit_dict(reference_ds, match_ds, ref_size, occlusion_mask=mask)
    plausiblility.save_plausibilities(fits, dataset_name="ycb", identifier="_{}_{}".format(shard, TOTAL_SHARDS))


    print("Finished computing plausibilities for shard {}/{}".format(shard, TOTAL_SHARDS))


def combine_shards():
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for sharded plausibility computation")
    parser.add_argument('shard', type=int)
    parser.add_argument('--combine_shards',
                        help='if passed this option, no plausibles will be '
                             'computed and all processed shards will be combined',
                        action='store_true')

    args = parser.parse_args()

    if args.combine_shards:
        combine_shards()
    else:
        compute_plausibles_for_shard(args.shard)
