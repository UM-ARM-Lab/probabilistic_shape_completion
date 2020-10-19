#! /usr/bin/env python
from shape_completion_training.utils import shapenet_storage

if __name__ == "__main__":
    shapenet_storage.write_shapenet_to_filelist(test_ratio=0.15,
                                                shape_ids=shapenet_storage.shapenet_labels(["mug"]))
