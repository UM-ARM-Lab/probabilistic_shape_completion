#! /usr/bin/env python

import unittest
import shape_completion_training.voxelgrid.metrics as metrics
import tensorflow as tf
import numpy as np
from setup_data_for_unit_tests import load_test_files
from shape_completion_training.voxelgrid import fit



class TestMetrics(unittest.TestCase):
    def test_iou_on_simple_data(self):
        vg1 = tf.random.uniform(shape=(10, 10, 10), minval=0.0, maxval=1.0)
        vg2 = 1.0 - vg1

        self.assertEqual(metrics.iou(vg1, vg1), 1.0)
        self.assertEqual(metrics.iou(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10, 10, 10))
        vg_half = tf.concat([tf.ones(shape=(10, 10, 5)), tf.zeros(shape=(10, 10, 5))], axis=2)
        self.assertEqual(metrics.iou(vg_all, vg_half), 0.5)

    def test_iou_on_shapes(self):
        d = load_test_files()
        self.assertEqual(metrics.iou(d[0], d[0]), 1.0)
        self.assertLess(metrics.iou(d[0], d[1]), 1.0)
        self.assertEqual(metrics.iou(d[0], d[1]), metrics.iou(d[1], d[0]))

    def test_p_correct(self):
        vg1 = tf.cast(tf.random.uniform(shape=(10, 10, 10), minval=0.0, maxval=1.0) > 0.5, tf.float32)
        vg2 = 1.0 - vg1

        self.assertEqual(metrics.p_correct(vg1, vg1), 1.0)
        self.assertEqual(metrics.p_correct(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10, 10, 10))
        vg_half = tf.concat([tf.ones(shape=(10, 10, 5)), 0.99 * tf.ones(shape=(10, 10, 5))], axis=2)
        self.assertAlmostEqual(metrics.p_correct(vg_half, vg_all), 0.99 ** (5 * 10 * 10), delta=1e-5)

    def test_p_correct_geometric_mean(self):
        vg1 = tf.cast(tf.random.uniform(shape=(10, 10, 10), minval=0.0, maxval=1.0) > 0.5, tf.float32)
        vg2 = 1.0 - vg1

        self.assertEqual(metrics.p_correct_geometric_mean(vg1, vg1), 1.0)
        self.assertEqual(metrics.p_correct_geometric_mean(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10, 10, 10))
        vg_half = tf.concat([tf.ones(shape=(10, 10, 5)), 0.5 * tf.ones(shape=(10, 10, 5))], axis=2)
        self.assertAlmostEqual(metrics.p_correct_geometric_mean(vg_half, vg_all), np.sqrt(.5), delta=1e-5)

    def test_highest_match(self):
        """
        Note that this test assumes that metric(a,b) is maximized when a==b
        """
        d = load_test_files()
        for i in range(3):
            test_vg = d[i]
            best_ind, best_elem = metrics.highest_match(test_vg, d)
            self.assertEqual(i, best_ind, "Wrong index returned")
            self.assertTrue((best_elem == d[i]).all(), "Returned element is not the best element")

    def test_chamfer_distance_pointcloud(self):
        a = tf.cast([[1.0, 0.0], [0.0, 0.0]], tf.float32)
        b = tf.cast([[2.0, 0.0], [4.0, 0.0], [10.0, 0.0]], tf.float32)
        d = (1 + 2.0) / 2 + (1 + 3 + 9.0) / 3
        self.assertEqual(d, metrics.chamfer_distance_pointcloud(a, b))
        self.assertEqual(d, metrics.chamfer_distance_pointcloud(b, a))
        self.assertEqual(0, metrics.chamfer_distance_pointcloud(a, a))

    def test_chamfer_distance_voxelgrid_on_simple(self):
        vg1 = tf.cast([[[0,0,0],[0,1,0], [0,1,0]],
                       [[0,0,0],[0,0,0], [0,0,0]],
                       [[0,0,0],[0,0,0], [0,0,0]]], tf.float32)
        vg2 = tf.cast([[[0,0,0],[0,1,0], [0,0,0]],
                       [[0,0,0],[0,1,0], [0,0,0]],
                       [[0,0,0],[0,0,0], [0,0,0]]], tf.float32)
        d = (0.0 + 1)/2 + (0.0 + 1)/2
        self.assertEqual(d, metrics.chamfer_distance(vg1, vg2, scale=1.0))
        self.assertEqual(d/4, metrics.chamfer_distance(vg1, vg2, scale=1.0/4))

    def test_chamfer_distance_voxelgrid_on_real(self):
        d = load_test_files()
        self.assertEqual(0.0, metrics.chamfer_distance(d[0], d[0], scale=0.1))
        self.assertGreater(metrics.chamfer_distance(d[0], d[1], scale=0.1), 0.0)

    def test_chamfer_is_better_after_fit(self):
        d = load_test_files()
        dist_orig = metrics.chamfer_distance(d[0], d[1], scale=0.1)
        vg_fit = fit.icp(d[1], d[0], scale=0.1)
        dist_fit = metrics.chamfer_distance(d[0], vg_fit, scale=0.1)
        self.assertLess(dist_fit, dist_orig, "Distance was not lower after fitting with icp")

    def test_highest_match_using_fit_and_chamfer_distance(self):
        d = load_test_files()
        base = d[0]

        def m(vg1, vg2):
            vg_fit = fit.icp(vg2, vg1, scale=0.1, max_iter=10, downsample=1)
            return -metrics.chamfer_distance(vg1, vg_fit, scale=0.1)

        ind, elem = metrics.highest_match(base, d, m)
        self.assertEqual(0, ind)

    def test_highest_match_using_fit_and_iou(self):
        d = load_test_files()
        base = d[0]

        def m(vg1, vg2):
            vg_fit = fit.icp(vg2, vg1, scale=0.1, max_iter=10, downsample=1)
            return metrics.iou(vg1, vg_fit)

        ind, elem = metrics.highest_match(base, d, m)
        self.assertEqual(0, ind)


class TestUtils(unittest.TestCase):
    def test_distance_matrix_on_simple_features(self):
        a = tf.cast([[1.0, 1.0], [0.0, 0.0]], tf.float32)
        b = tf.cast([[1.0, 2.0], [4.0, 5.0], [10.0, 1.0]], tf.float32)
        d = tf.cast([[1, 5, 9], [np.sqrt(5), np.sqrt(41), np.sqrt(101)]], tf.float32)
        self.assertTrue(tf.math.equal(d, metrics.distance_matrix(a, b)).numpy().all())
        # self.assertEqual(d, metrics.distance_matrix(a,b))


if __name__ == '__main__':
    unittest.main()
