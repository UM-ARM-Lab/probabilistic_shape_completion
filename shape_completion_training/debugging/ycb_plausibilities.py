#! /usr/bin/env python
from shape_completion_training.model import plausiblility
from shape_completion_training.utils import data_tools
import argparse



"""
Plausibilities for the YCB dataset with slit occlusion
"""

if __name__ == "__main__":

    # test_dataset = data_tools._load_metadata_train_or_test(shapes="all", shuffle=False, prefix="test")
    _, ds = data_tools.load_dataset("ycb", shuffle=False, metadata_only=True)
    # sharded_test_dataset = ds.shard(TOTAL_SHARDS, args.shard)
    # sub_ds = ds.take(2)

    params = {'apply_slit_occlusion': True}

    ref_size = 0
    for _ in ds:
        ref_size += 1

    ds = data_tools.load_voxelgrids(ds)

    reference_ds = data_tools.preprocess_test_dataset(ds, params)

    single_match_ds = data_tools.simulate_input(ds, 0, 0, 0, sim_input_fn=data_tools.simulate_2_5D_input)
    match_ds =  data_tools.apply_fixed_slit_occlusion(single_match_ds, 32, 6)

    for slit_min in range(32-10, 32+10, 2):
        if slit_min == 32:
            continue
        match_ds = match_ds.concatenate(data_tools.apply_fixed_slit_occlusion(single_match_ds, slit_min, 6))

    mask = data_tools.get_slit_occlusion_2D_mask(28, 6, (64, 64))

    fits = plausiblility.compute_partial_icp_fit_dict(reference_ds, match_ds, ref_size, occlusion_mask=mask)
    plausiblility.save_plausibilities(fits, dataset_name="ycb")
    #
    # loaded_fits = plausiblility.load_plausibilities()
    print("Finished computing plausibilities")
