#! /usr/bin/env python

import unittest
import voxelgrid_tools as vt
import tensorflow as tf
import numpy as np



class TestVoxelgridTools(unittest.TestCase):
    def test_iou(self):
        vg1 = tf.random.uniform(shape=(10,10,10), minval=0.0, maxval=1.0)
        vg2 = 1.0 - vg1
        
        self.assertEqual(vt.iou(vg1, vg1), 1.0)
        self.assertEqual(vt.iou(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10,10,10))
        vg_half = tf.concat([tf.ones(shape=(10,10,5)), tf.zeros(shape=(10,10,5))], axis=2)
        self.assertEqual(vt.iou(vg_all, vg_half), 0.5)


    def test_p_correct(self):
        vg1 = tf.cast(tf.random.uniform(shape=(10,10,10), minval=0.0, maxval=1.0) > 0.5, tf.float32)
        vg2 = 1.0 - vg1
        
        self.assertEqual(vt.p_correct(vg1, vg1), 1.0)
        self.assertEqual(vt.p_correct(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10,10,10))
        vg_half = tf.concat([tf.ones(shape=(10,10,5)), 0.99*tf.ones(shape=(10,10,5))], axis=2)
        self.assertAlmostEqual(vt.p_correct(vg_half, vg_all), 0.99**(5*10*10), delta=1e-5)

    def test_p_correct_geometric_mean(self):
        vg1 = tf.cast(tf.random.uniform(shape=(10,10,10), minval=0.0, maxval=1.0) > 0.5, tf.float32)
        vg2 = 1.0 - vg1
        
        self.assertEqual(vt.p_correct_geometric_mean(vg1, vg1), 1.0)
        self.assertEqual(vt.p_correct_geometric_mean(vg1, vg2), 0.0)

        vg_all = tf.ones(shape=(10,10,10))
        vg_half = tf.concat([tf.ones(shape=(10,10,5)), 0.5*tf.ones(shape=(10,10,5))], axis=2)
        self.assertAlmostEqual(vt.p_correct_geometric_mean(vg_half, vg_all), np.sqrt(.5), delta=1e-5)


if __name__ == '__main__':
    unittest.main()
