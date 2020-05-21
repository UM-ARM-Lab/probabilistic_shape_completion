from unittest import TestCase
from setup_data_for_unit_tests import load_test_files
from shape_completion_training.voxelgrid.conversions import voxelgrid_to_pointcloud, pointcloud_to_voxelgrid, \
    transform_voxelgrid, make_transform, downsample, format_voxelgrid
import numpy as np
import tensorflow as tf


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

    def test_transform_voxelgrid_using_identity(self):
        vg_orig = load_test_files()[0]
        T = np.eye(4)
        vg_new = transform_voxelgrid(vg_orig, T)
        self.assertTrue((vg_new == vg_orig).all())

    def test_transform_voxelgrid_is_invertable(self):
        """
        Note, this is not necessarily true for rotation or fractional translation, as converting to a voxelgrid
        effectively truncates to an int so there is a loss of precision
        @return:
        """
        T = make_transform(thetas=[0.0, 0, np.pi / 2], translation=[1, 2, 3])
        T_inv = np.linalg.inv(T)
        scale = 0.5
        self.assertTrue((np.dot(T, T_inv) == np.eye(4)).all())
        vg_orig = load_test_files()[0]
        vg_rot = transform_voxelgrid(vg_orig, T, scale=scale)
        self.assertFalse((vg_orig == vg_rot).all())
        vg_new = transform_voxelgrid(vg_rot, T_inv, scale=scale)
        self.assertTrue((vg_orig == vg_new).all())


class TestUtils(TestCase):
    def test_format_keeps_numpy_as_numpy(self):
        vg = np.random.random((3, 3, 3))
        vg_formatted = format_voxelgrid(vg, leading_dim=False, trailing_dim=False)
        self.assertTrue(isinstance(vg_formatted, np.ndarray))
        self.assertFalse(tf.is_tensor(vg_formatted))
        self.assertEqual((3, 3, 3), vg_formatted.shape)
        self.assertTrue((vg == vg_formatted).all())

    def test_format_keeps_tf_as_tf(self):
        vg = tf.random.uniform((3, 3, 3))
        vg_formatted = format_voxelgrid(vg, leading_dim=False, trailing_dim=False)
        self.assertFalse(isinstance(vg_formatted, np.ndarray))
        self.assertTrue(tf.is_tensor(vg_formatted))
        self.assertEqual((3, 3, 3), vg_formatted.shape)
        self.assertTrue(tf.reduce_all(vg == vg_formatted))

    def test_format_adds_and_removes_tf_dims(self):
        vg_neither = tf.random.uniform((3, 3, 3))
        vg_leading = format_voxelgrid(vg_neither, leading_dim=True, trailing_dim=False)
        vg_trailing = format_voxelgrid(vg_neither, leading_dim=False, trailing_dim=True)
        vg_both = format_voxelgrid(vg_neither, leading_dim=True, trailing_dim=True)
        self.assertEqual((1, 3, 3, 3), vg_leading.shape)
        self.assertEqual((3, 3, 3, 1), vg_trailing.shape)
        self.assertEqual((1, 3, 3, 3, 1), vg_both.shape)

        for vg in [vg_neither, vg_leading, vg_trailing, vg_both]:
            self.assertEqual((3, 3, 3), format_voxelgrid(vg, False, False).shape)
            self.assertEqual((1, 3, 3, 3), format_voxelgrid(vg, True, False).shape)
            self.assertEqual((3, 3, 3, 1), format_voxelgrid(vg, False, True).shape)
            self.assertEqual((1, 3, 3, 3, 1), format_voxelgrid(vg, True, True).shape)

    def test_downsample_without_extra_dims(self):
        vg_orig = np.zeros((4, 4, 4))
        vg_orig[0, 0, 0] = 1.0
        vg_orig = tf.cast(vg_orig, tf.float32)
        vg_downsampled = downsample(vg_orig)
        self.assertEqual((2, 2, 2), vg_downsampled.shape)
        vg_expected = [[[1, 0], [0, 0]], [[0, 0], [0, 0]]]
        self.assertTrue(tf.reduce_all(vg_expected == vg_downsampled))

    def test_downsample_with_extra_dims(self):
        vg_orig = np.zeros((1, 4, 4, 4))
        vg_orig[0, 0, 0, 0] = 1.0
        self.assertEqual((1,2,2,2), downsample(vg_orig).shape)

        vg_orig = np.zeros((1, 4, 4, 4, 1))
        vg_orig[0, 0, 0, 0] = 1.0
        self.assertEqual((1, 2, 2, 2, 1), downsample(vg_orig).shape)

        vg_orig = np.zeros((4, 4, 4, 1))
        vg_orig[0, 0, 0, 0] = 1.0
        self.assertEqual((2, 2, 2, 1), downsample(vg_orig).shape)
