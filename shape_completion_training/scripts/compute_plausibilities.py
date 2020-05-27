#! /usr/bin/env python
from shape_completion_training.model import plausiblility
from shape_completion_training.model import data_tools

if __name__ == "__main__":
    train_dataset, test_dataset = data_tools.load_shapenet_metadata(shuffle=False)
    # dataset = dataset.take(100)

    fits = plausiblility.compute_icp_fit_dict(test_dataset)
    plausiblility.save_plausibilities(fits)

    loaded_fits = plausiblility.load_plausibilities()
    print("Finished computing plausibilities")
