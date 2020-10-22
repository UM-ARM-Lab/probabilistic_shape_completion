from unittest import TestCase
import tensorflow as tf
from shape_completion_training.utils import tf_utils


class TestUtils(TestCase):
    def test_geometric_mean(self):
        t = tf.convert_to_tensor([[1, 3, 9], [1, 1, 27.]])
        self.assertEqual(3, tf_utils.reduce_geometric_mean(t))
