#! /usr/bin/env python

import unittest
import shape_completion_training.model.metrics as metrics
import tensorflow as tf
import numpy as np
from setup_data_for_unit_tests import load_test_files


class TestMetrics(unittest.TestCase):
    def test_iou_on_simple_data(self):
        vg1 = tf.random.uniform(shape=(10,10,10), minval=0.0, maxval=1.0)
        vg2 = 1.0 - vg1
        
        self.assertEqual(metrics.iou(vg1, vg1), 1.0)
        self.assertEqual(metrics.iou(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10,10,10))
        vg_half = tf.concat([tf.ones(shape=(10,10,5)), tf.zeros(shape=(10,10,5))], axis=2)
        self.assertEqual(metrics.iou(vg_all, vg_half), 0.5)

    def test_iou_on_shapes(self):
        d = load_test_files()
        self.assertEqual(metrics.iou(d[0], d[0]), 1.0)
        self.assertLess(metrics.iou(d[0], d[1]), 1.0)
        self.assertEqual(metrics.iou(d[0], d[1]), metrics.iou(d[1], d[0]))

    def test_p_correct(self):
        vg1 = tf.cast(tf.random.uniform(shape=(10,10,10), minval=0.0, maxval=1.0) > 0.5, tf.float32)
        vg2 = 1.0 - vg1
        
        self.assertEqual(metrics.p_correct(vg1, vg1), 1.0)
        self.assertEqual(metrics.p_correct(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10,10,10))
        vg_half = tf.concat([tf.ones(shape=(10,10,5)), 0.99*tf.ones(shape=(10,10,5))], axis=2)
        self.assertAlmostEqual(metrics.p_correct(vg_half, vg_all), 0.99 ** (5 * 10 * 10), delta=1e-5)

    def test_p_correct_geometric_mean(self):
        vg1 = tf.cast(tf.random.uniform(shape=(10,10,10), minval=0.0, maxval=1.0) > 0.5, tf.float32)
        vg2 = 1.0 - vg1
        
        self.assertEqual(metrics.p_correct_geometric_mean(vg1, vg1), 1.0)
        self.assertEqual(metrics.p_correct_geometric_mean(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10,10,10))
        vg_half = tf.concat([tf.ones(shape=(10,10,5)), 0.5*tf.ones(shape=(10,10,5))], axis=2)
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


if __name__ == '__main__':
    unittest.main()
