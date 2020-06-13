#! /usr/bin/env python

from shape_completion_training.model import data_tools
from shape_completion_training.model.modelrunner import ModelRunner

params = {
    'batch_size': 1500,
    'network': 'RealNVP',
    'dim': 24,
    'num_masked': 12,
    'learning_rate': 1e-5,
}

if __name__ == "__main__":
    sn = data_tools.get_shapenet()

    mr = ModelRunner(training=True, params=params, group_name="Flow")
    # IPython.embed()

    mr.train_and_test(sn.train_ds)
