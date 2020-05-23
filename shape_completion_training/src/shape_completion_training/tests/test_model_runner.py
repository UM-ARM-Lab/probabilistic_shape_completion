import unittest

import tensorflow as tf

from shape_completion_training.model.utils import reduce_mean_dict
from shape_completion_training.modelrunner import ModelRunner
from shape_completion_training.mykerasmodel import MyKerasModel

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
    'learning_rate': 0.001,
}


class FakeModel(MyKerasModel):
    def __init__(self, params, batch_size=16):
        super().__init__(hparams=params, batch_size=batch_size)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, dataset_element, training=False, **kwargs):
        inputs, _ = dataset_element
        x = inputs['x']
        y = self.dense(x)
        return {'y': y}

    @tf.function
    def compute_loss(self, dataset_element, outputs):
        loss = tf.keras.losses.mse(dataset_element[1]['y'], outputs['y'])
        losses = {
            "loss": loss,
            "mock_loss": tf.constant(3),
        }
        return reduce_mean_dict(losses)


class ModelRunnerTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        numbers = tf.data.Dataset.range(1000).map(lambda x: tf.cast(x, tf.float32))
        x = numbers.map(lambda x: {'x': x}).batch(1)
        y = numbers.map(lambda x: {'y': x * 5 + 2}).batch(1)
        cls.dataset = tf.data.Dataset.zip((x, y))

    def test_train(self):
        model = FakeModel(params=params)
        mr = ModelRunner(model=model, params=params, write_summary=False)
        mr.train(ModelRunnerTraining.dataset, num_epochs=1)

    def test_train_named(self):
        model = FakeModel(params=params)
        mr = ModelRunner(model=model, params=params, write_summary=False, trial_name='test_name')
        mr.train(ModelRunnerTraining.dataset, num_epochs=1)


if __name__ == '__main__':
    unittest.main()
