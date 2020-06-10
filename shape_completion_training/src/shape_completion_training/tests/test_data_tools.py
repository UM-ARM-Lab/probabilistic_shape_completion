from unittest import TestCase
from  shape_completion_training.model import data_tools, filepath_tools
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


class TestSavingAndLoading(TestCase):
    """
    This fail for multiple reasons now:
    Unittess are not setup to run like ROS
    the path name is meaningful now, so "/tmp" is not valid
    """
    def _test_save_and_reload_does_not_change_values(self):
        gt = (np.random.random((64,64,64,1)) > 0.5).astype(float)
        self.assertEqual(1.0, np.max(gt))
        self.assertEqual(0.0, np.min(gt))


        data_tools.save_gt_voxels("/tmp", "0", gt)
        reloaded = data_tools.load_gt_voxels("/tmp", "0")
        self.assertTrue(np.all(reloaded['gt_occ'] == gt))
        self.assertFalse("gt_occ_packed" in reloaded)
