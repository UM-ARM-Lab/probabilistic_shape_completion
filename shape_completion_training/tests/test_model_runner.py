import unittest

import tensorflow as tf

from shape_completion_training.model.modelrunner import ModelRunner

params = {
    'num_latent_layers': 200,
    'translation_pixel_range_x': 10,
    'translation_pixel_range_y': 10,
    'translation_pixel_range_z': 10,
    'is_u_connected': False,
    'final_activation': 'None',
    'unet_dropout_rate': 0.5,
    'use_final_unet_layer': False,
    'simulate_partial_completion': False,
    'simulate_random_partial_completion': False,
    # 'network': 'VoxelCNN',
    # 'network': 'VAE_GAN',
    # 'network': 'Augmented_VAE',
    # 'network': 'Conditional_VCNN',
    'network': 'AE_VCNN',
    'stacknet_version': 'v2',
    'turn_on_prob': 0.00000,
    'turn_off_prob': 0.0,
    'loss': 'cross_entropy',
    'multistep_loss': False,
}


class ModelRunnerTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        size = 64
        range = tf.data.Dataset.range(size * size * size)
        cls.dataset = tf.data.Dataset

    def test_train(self):
        ModelRunner(training=True, params=params, write_summary=False)


if __name__ == '__main__':
    unittest.main()
