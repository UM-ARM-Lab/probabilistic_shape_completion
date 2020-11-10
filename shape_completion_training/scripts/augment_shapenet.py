#! /usr/bin/env python
from __future__ import print_function

import shape_completion_training.utils.shapenet_storage
from shape_completion_training.utils import shapenet_storage
import datetime
import rospy
from shape_completion_training.utils.data_augmentation import NUM_THREADS, augment_category

# HARDCODED_BOUNDARY = '-bb -0.6 -0.6 -0.6 0.6 0.6 0.6'

"""
NOTE:
If running over ssh, need to start a virtual screen
https://www.patrickmin.com/binvox/

Xvfb :99 -screen 0 640x480x24 &
export DISPLAY=:99


Then run binvox with the -pb option 

"""

# def augment_single(basepath):
#     """
#     Augment a hardcoded single shape. Useful for debugging
#     """
#     shape_id = 'a1d293f5cc20d01ad7f470ee20dce9e0'
#     fp = basepath / shape_id / 'models'
#     print("Augmenting single models at {}".format(fp))
#
#     old_files = [f for f in fp.iterdir() if f.name.startswith("model_augmented")]
#     for f in old_files:
#         f.unlink()
#
#     obj_path = fp / "model_normalized.obj"
#     obj_tools.augment(obj_path.as_posix())
#
#     augmented_obj_files = [f for f in fp.iterdir()
#                            if f.name.startswith('model_augmented')
#                            if f.name.endswith('.obj')]
#     augmented_obj_files.sort()
#     for f in augmented_obj_files:
#         binvox_object_file(f)


if __name__ == "__main__":
    rospy.init_node("augment_shapenet_node")
    sn_path = shapenet_storage.shapenet_load_path
    # sn_path = sn_path / shape_completion_training.utils.shapenet_storage.shape_map['mug']
    sn_path = sn_path / shape_completion_training.utils.shapenet_storage.shape_map['table']

    start_time = datetime.datetime.now()

    # augment_category(sn_path)
    augment_category(sn_path, shape_ids = ["fff7f07d1c4042f8a946c24c4f9fb58e"])
    print("")
    print("Augmenting with {} threads took {} seconds".format(NUM_THREADS, datetime.datetime.now() - start_time))
