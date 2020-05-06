from copy import deepcopy

import tensorflow as tf


class MyModel(tf.keras.Model):

    def __init__(self, hparams, batch_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.hparams = deepcopy(hparams)
        self.optimizer = tf.keras.optimizers.Adam(hparams['learning_rate'])

    def call(self, inputs, training=False, **kwargs):
        raise NotImplementedError()

    def compute_loss(self, train_element, train_predictions):
        raise NotImplementedError()

    @tf.function
    def apply_gradients(self, tape, train_element, train_predictions, losses):
        train_batch_loss = losses['loss']
        variables = self.trainable_variables
        gradients = tape.gradient(train_batch_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

    @tf.function
    def train_step(self, train_element):
        all_metrics = {}
        with tf.GradientTape() as tape:
            train_predictions = self.call(train_element, training=True)
            train_losses, loss_metrics = self.loss_function(train_element, train_predictions)

        gradient_metrics = self.apply_gradients(tape, train_element, train_predictions, train_losses)

        all_metrics.update(loss_metrics)
        all_metrics.update(gradient_metrics)

        return train_predictions, all_metrics

    def calculate_metrics(self, train_elements, train_predictions):
        raise NotImplementedError()
