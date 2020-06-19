#!/usr/bin/env python
from __future__ import print_function

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np
from time import time
from shape_completion_training.model import mykerasmodel

tfd = tfp.distributions
tfb = tfp.bijectors

settings = {
    # 'batch_size': 1500,
    # 'method': 'NVP',
    'num_bijectors': 8,
    'train_iters': 2e5,
    'visualize_data': False,
}


class Flow(mykerasmodel.MyKerasModel):
    def __init__(self, *args, **kwargs):
        super(Flow, self).__init__(*args, **kwargs)
        flow = None

    def call(self, *inputs):
        return self.flow.bijector.forward(*inputs)

    def build_model(self,):
        x = self.flow.distribution.sample(self.batch_size)
        r = self.flow.bijector.forward(x)
        self.built = True
        return r

    # @tf.function
    def train_step(self, elem):
        X = tf.keras.layers.Flatten()(tf.cast(elem['bounding_box'], tf.float32))
        with tf.GradientTape() as tape:
            loss = -tf.reduce_mean(self.flow.log_prob(X, training=True))
            gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {'loss': loss}


class MAF(Flow):
    def __init__(self, output_dim, num_masked, **kwargs):
        super(MAF, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked

        self.bijector_fns = []

        bijectors = []
        for i in range(settings['num_bijectors']):
            self.bijector_fns.append(tfb.masked_autoregressive_default_template(hidden_layers=[512, 512]))
            bijectors.append(
                tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=self.bijector_fns[-1]
                )
            )

            # if i%2 == 0:
            #     bijectors.append(tfb.BatchNormalization())

            bijectors.append(tfb.Permute(permutation=[1, 0]))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0, 0.0]),
            bijector=bijector)


class RealNVP(Flow):
    def __init__(self, training=False, *args, **kwargs):
        super(RealNVP, self).__init__(*args, **kwargs)


        self.dim = self.hparams['dim']
        self.num_masked = self.hparams['num_masked']

        self.bijector_fns = []
        self.bijector_fn = tfp.bijectors.real_nvp_default_template(hidden_layers=[512, 512])

        bijectors = []
        for i in range(settings['num_bijectors']):
            # Note: Must store the bijectors separately, otherwise only a single set of tf variables is created for all layers
            self.bijector_fns.append(tfp.bijectors.real_nvp_default_template(hidden_layers=[512, 512]))
            bijectors.append(
                tfb.RealNVP(num_masked=self.num_masked,
                            shift_and_log_scale_fn=self.bijector_fns[-1])
            )

            if i % 3 == 0:
                bijectors.append(tfb.BatchNormalization(training=training))

            permutation = [i for i in reversed(range(self.dim))]
            bijectors.append(tfb.Permute(permutation=permutation))

        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        # bijector = tfb.Chain(bijectors[:-1])

        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0.0] * self.dim),
            bijector=bijector)


# def train(model, ds, optimizer, print_period=1000):
#     """
#     Train `model` on dataset `ds` using optimizer `optimizer`,
#       prining the current loss every `print_period` iterations
#     """
#     start = time()
#     itr = ds.__iter__()
#     # for i in range(int(2e5 + 1)):
#     for i in range(int(settings['train_iters'] + 1)):
#         X = next(itr)
#         loss = model.train_step(X, optimizer).numpy()
#         if i % print_period == 0:
#             print("{} loss: {}, {}s".format(i, loss, time() - start))
#             if np.isnan(loss):
#                 break
#     return loss


# def print_settings():
#     """
#     display the settings used when creating the model
#     """
#     print("Using settings:")
#     for k in settings.keys():
#         print('{}: {}'.format(k, settings[k]))


def build_model(model):
    """
    Run a pass of the model to initialize the tensorflow network
    """
    x = model.flow.distribution.sample(8000)
    for bijector in reversed(model.flow.bijector.bijectors):
        x = bijector.forward(x)


def train_and_run_model(display=True):
    print_settings()

    ds, pts = create_dataset()

    if settings['method'] == 'MAF':
        model = MAF(output_dim=2, num_masked=1)
    elif settings['method'] == 'NVP':
        model = RealNVP(output_dim=2, num_masked=1)

    model(pts)
    build_model(model)
    if display:
        model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=settings['learning_rate'])
    loss = train(model, ds, optimizer)

    if display:
        XF = model.flow.sample(2000)

    return loss


if __name__ == "__main__":
    train_and_run_model()
    # run_statistics_trial()
