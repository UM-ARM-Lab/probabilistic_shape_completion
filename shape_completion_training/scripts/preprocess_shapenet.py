#! /usr/bin/env python
from shape_completion_training.model import data_tools

if __name__ == "__main__":
    data_tools.write_shapenet_to_tfrecord(test_ratio=0.02, shape_ids=data_tools.shapenet_labels(["mug"]))
