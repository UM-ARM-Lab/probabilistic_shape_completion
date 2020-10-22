#! /usr/bin/env python
from shape_completion_training.plausible_diversity import plausiblility
from shape_completion_training.utils import data_tools

if __name__ == "__main__":

    train_dataset, test_dataset = data_tools.load_shapenet_metadata(shuffle=False)
    # dataset = dataset.take(100)

    params = {'apply_slit_occlusion': False}

    fits = plausiblility.compute_icp_fit_dict(test_dataset, params)
    plausiblility.save_plausibilities(fits)

    loaded_fits = plausiblility.load_plausibilities()
    print("Finished computing plausibilities")
