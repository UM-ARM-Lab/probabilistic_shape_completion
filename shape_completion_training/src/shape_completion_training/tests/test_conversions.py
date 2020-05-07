from unittest import TestCase
from setup_data_for_unit_tests import load_test_files
from shape_completion_training.voxelgrid.conversions import voxelgrid_to_pointcloud, pointcloud_to_voxelgrid
import numpy as np


class TestConversions(TestCase):
    def test_voxelgrid_to_pointcloud_on_simple(self):
        vg_orig = np.zeros([3, 3, 3])
        pt_cloud = voxelgrid_to_pointcloud(vg_orig)
        vg_new = pointcloud_to_voxelgrid(pt_cloud, add_leading_dim=True, add_trailing_dim=True, shape=[3, 3, 3])
        self.assertTrue((vg_orig == vg_new).all())

    def test_voxelgrid_to_pointcloud_with_scaling_on_simple(self):
        vg_orig = np.zeros([3, 3, 3])
        for scale in [0.02, 1.0, 10., 10000.]:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig, scale=scale)
            vg_new = pointcloud_to_voxelgrid(pt_cloud, scale=scale,
                                             add_leading_dim=True, add_trailing_dim=True, shape=[3, 3, 3])
            self.assertTrue((vg_orig == vg_new).all())

    def test_voxelgrid_to_pointcloud_with_scaling(self):
        vg_orig = load_test_files()[0]
        for scale in [0.01, 1.0, 10., 10000.]:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig, scale=scale)
            vg_new = pointcloud_to_voxelgrid(pt_cloud, scale=scale,
                                             add_leading_dim=True, add_trailing_dim=True, shape=(64, 64, 64))
            self.assertTrue((vg_orig == vg_new).all(), "Failed on scaling {}".format(scale))

    def test_voxelgrid_to_pointcloud_with_scaling_and_translation(self):
        vg_orig = load_test_files()[0]
        for scale in [0.01, 1.0, 10., 10000.]:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig, scale=scale)
            pt_cloud += scale/4
            vg_new = pointcloud_to_voxelgrid(pt_cloud, scale=scale,
                                             add_leading_dim=True, add_trailing_dim=True, shape=(64, 64, 64))
            self.assertTrue((vg_orig == vg_new).all(), "Failed on scaling {}".format(scale))

    def test_voxelgrid_to_pointcloud_on_large_data(self):
        d = load_test_files()
        for vg_orig in d:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig)
            vg_new = pointcloud_to_voxelgrid(pt_cloud, add_leading_dim=True, add_trailing_dim=True)
            self.assertTrue((vg_orig == vg_new).all())
