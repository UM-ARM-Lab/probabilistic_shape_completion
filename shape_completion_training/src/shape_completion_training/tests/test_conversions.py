from unittest import TestCase
from setup_data_for_unit_tests import load_test_files
from shape_completion_training.voxelgrid.conversions import voxelgrid_to_pointcloud, pointcloud_to_voxelgrid, \
    transform_voxelgrid
import numpy as np


class TestConversions(TestCase):
    def test_voxelgrid_to_pointcloud_on_simple(self):
        vg_orig = np.zeros([3, 3, 3])
        vg_orig[(1, 2), (0, 2), (1, 0)] = 1.0
        pt_cloud = voxelgrid_to_pointcloud(vg_orig)
        vg_new = pointcloud_to_voxelgrid(pt_cloud, add_leading_dim=False, add_trailing_dim=False, shape=[3, 3, 3])
        self.assertTrue((vg_orig == vg_new).all())

    def test_voxelgrid_to_pointcloud_with_scaling_on_simple(self):
        vg_orig = np.zeros([3, 3, 3])
        vg_orig[(1, 2), (0, 2), (1, 0)] = 1.0
        for scale in [0.02, 1.0, 10., 10000.]:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig, scale=scale)
            vg_new = pointcloud_to_voxelgrid(pt_cloud, scale=scale,
                                             add_leading_dim=False, add_trailing_dim=False, shape=[3, 3, 3])
            self.assertTrue((vg_orig == vg_new).all())

    def test_voxelgrid_to_pointcloud_with_scaling(self):
        vg_orig = load_test_files()[0]
        for scale in [0.01, 1.0, 10., 10000.]:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig, scale=scale)
            vg_new = pointcloud_to_voxelgrid(pt_cloud, scale=scale,
                                             add_leading_dim=True, add_trailing_dim=True, shape=(64, 64, 64))
            self.assertTrue((vg_orig == vg_new).all(), "Failed on scaling {}".format(scale))

    def test_voxelgrid_to_pointcloud_with_scaling_and_origin(self):
        vg_orig = load_test_files()[0]
        origin = [120, -12, 130]
        for scale in [0.01, 1.0, 10., 10000.]:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig, scale=scale, origin=origin)
            vg_new = pointcloud_to_voxelgrid(pt_cloud, scale=scale, origin=origin,
                                             add_leading_dim=True, add_trailing_dim=True, shape=(64, 64, 64))
            self.assertTrue((vg_orig == vg_new).all(), "Failed on scaling {}".format(scale))

    def test_voxelgrid_to_pointcloud_with_scaling_and_translation(self):
        vg_orig = load_test_files()[0]
        for scale in [0.01, 1.0, 10., 10000.]:
            pt_cloud = voxelgrid_to_pointcloud(vg_orig, scale=scale)
            pt_cloud += scale / 4
            vg_new = pointcloud_to_voxelgrid(pt_cloud, scale=scale,
                                             add_leading_dim=True, add_trailing_dim=True, shape=(64, 64, 64))
            self.assertTrue((vg_orig == vg_new).all(), "Failed on scaling {}".format(scale))

    def test_voxelgrid_to_pointcloud_on_large_data(self):
        d = load_test_files()
        for i, vg_orig in enumerate(d):
            pt_cloud = voxelgrid_to_pointcloud(vg_orig)
            vg_new = pointcloud_to_voxelgrid(pt_cloud, add_leading_dim=True, add_trailing_dim=True)
            self.assertTrue((vg_orig == vg_new).all(), "Failed on point cloud {}".format(i))


class TestTransforms(TestCase):
    def test_transform_voxelgrid_on_empty(self):
        vg_orig = np.zeros([3, 3, 3])
        T = np.eye(4)
        vg_new = transform_voxelgrid(vg_orig, T)
        self.assertTrue((vg_new == vg_orig).all())

    def test_transform_voxelgrid_using_identity_on_simple_data(self):
        vg_orig = np.zeros([3, 3, 3])
        vg_orig[(1, 2), (0, 2), (1, 0)] = 1.0
        T = np.eye(4)
        vg_new = transform_voxelgrid(vg_orig, T)
        self.assertTrue((vg_new == vg_orig).all())
