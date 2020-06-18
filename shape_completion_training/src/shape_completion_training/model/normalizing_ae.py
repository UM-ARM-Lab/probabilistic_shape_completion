import tensorflow as tf
from shape_completion_training.model.mykerasmodel import MyKerasModel
import tensorflow.keras.layers as tfl

from shape_completion_training.model.utils import stack_known, log_normal_pdf


class NormalizingAE(MyKerasModel):
    def __init__(self, hparams, batch_size, *args, **kwargs):
        super(NormalizingAE, self).__init__(hparams, batch_size, *args, **kwargs)
        self.flow = None
        self.encoder = make_encoder(inp_shape=[64, 64, 64, 2], params=hparams)

    def call(self, dataset_element, training=False, **kwargs):
        known = stack_known(dataset_element)
        mean, logvar = self.encode(known)
        sampled_features = self.sample_latent(mean, logvar)
        return {"mean": mean, "logvar": logvar, "sampled_features": sampled_features}

    def compute_loss(self, train_element, train_outputs):
        bb = tf.keras.layers.Flatten()(tf.cast(train_element['bounding_box'], tf.float32))
        expected = self.flow.bijector.inverse(bb)
        losses = -log_normal_pdf(expected, train_outputs['mean'], train_outputs['logvar'])
        loss = tf.reduce_mean(losses)
        return {"loss": loss}

    def sample_latent(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        features = eps * tf.exp(logvar * 0.5) + mean
        return features

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def predict(self, elem):
        return self(next(elem.__iter__()))


def make_encoder(inp_shape, params):
    """Basic VAE encoder"""
    n_features = params['num_latent_layers']

    return tf.keras.Sequential(
        [
            tfl.InputLayer(input_shape=inp_shape),

            tfl.Conv3D(64, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Conv3D(128, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Conv3D(256, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Conv3D(512, (2, 2, 2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2, 2, 2)),

            tfl.Flatten(),
            tfl.Dense(n_features * 2)
        ]
    )
