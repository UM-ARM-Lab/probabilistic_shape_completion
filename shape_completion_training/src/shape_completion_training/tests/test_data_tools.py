from unittest import TestCase
import shape_completion_training.model.data_tools as data_tools
import tensorflow as tf
import numpy as np
from shape_completion_training.voxelgrid import conversions


class TestDatasetLoading(TestCase):
    def test_dataset_exists_and_can_be_loaded(self):
        train_data_shapenet, test_data_shapenet = data_tools.load_shapenet([data_tools.shape_map["mug"]])


class TestObservationModel(TestCase):
    def test_get_depth_map(self):
        vg = np.zeros((3,3,3))
        vg[1,1,1] = 1.0
        vg[2,1,1] = 1.0
        vg[1,1,0] = 1.0
        vg = tf.cast(vg, tf.float32)
        img = data_tools.simulate_depth_image(vg)
        vg_2_5D, _ = data_tools.simulate_2_5D_input(conversions.format_voxelgrid(vg, False, True).numpy())
        img_from_2_5D = data_tools.simulate_depth_image(vg_2_5D)
        self.assertTrue((img == img_from_2_5D).numpy().all())
