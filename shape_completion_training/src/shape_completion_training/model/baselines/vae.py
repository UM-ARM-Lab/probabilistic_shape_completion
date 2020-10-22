import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tfl

import shape_completion_training.model.nn_tools as nn
from shape_completion_training.utils.tf_utils import stack_known
from shape_completion_training.model.mykerasmodel import MyKerasModel


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_vae_loss(z, mean, logvar, sample_logit, labels):
    """Computes vae loss given:
    z: latent space sample
    mean: mean of the feature space outputed by the encoder
    logvar: logvar of the feature space outputed by the endocer
    sample_logits: logits (before sigmoid) output of the decoder
    labels: ground truth voxels
    
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    sample_logit = model.decode(z)

    See https://tensorflow.org/tutorials/generative/cvae for more details
    """

    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logit, labels=labels)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


class VAE(MyKerasModel):
    def __init__(self, hparams, batch_size, *args, **kwargs):
        super(VAE, self).__init__(hparams, batch_size, *args, **kwargs)
        self.make_vae(inp_shape=[64, 64, 64, 2])

    def get_model(self):
        return self

    def make_vae(self, inp_shape):
        self.encoder = make_encoder(inp_shape, self.hparams)
        self.generator = make_generator(self.hparams)

    def predict(self, elem):
        return self(next(elem.__iter__()))

    def call(self, inp):
        known = stack_known(inp)
        mean, logvar = self.encode(known)
        z = self.reparameterize(mean, logvar)
        sample = self.decode(z, apply_sigmoid=True)
        output = {'predicted_occ': sample, 'predicted_free': 1 - sample}
        return output

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generator(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            known = stack_known(batch)
            mean, logvar = self.encode(known)
            z = self.reparameterize(mean, logvar)
            sample_logit = self.decode(z)
            vae_loss = compute_vae_loss(z, mean, logvar, sample_logit, labels=batch['gt_occ'])

            sample = tf.nn.sigmoid(sample_logit)
            output = {'predicted_occ': sample, 'predicted_free': 1 - sample}
            metrics = nn.calc_metrics(output, batch)

            vae_variables = self.encoder.trainable_variables + self.generator.trainable_variables
            gradients = tape.gradient(vae_loss, vae_variables)

        self.optimizer.apply_gradients(list(zip(gradients, vae_variables)))

        metrics = {k: tf.reduce_mean(metrics[k]) for k in metrics}
        metrics['loss'] = vae_loss
        return output, metrics


class VAE_GAN(VAE):
    def __init__(self, hparams, batch_size, *args, **kwargs):
        super(VAE_GAN, self).__init__(hparams, batch_size, *args, **kwargs)
        self.dis_opt = tf.keras.optimizers.Adam(hparams['discriminator_learning_rate'])
        self.discriminator = make_discriminator([64, 64, 64, 3], self.hparams)

    def discriminate(self, known_input, output):
        inp = tf.concat([known_input, output], axis=4)
        return self.discriminator(inp)

    def gradient_penalty(self, known, real, fake):
        alpha = tf.random.uniform([tf.shape(known)[0], 1, 1, 1, 1], 0.0, 1.0)
        diff = fake - real
        interp = real + (alpha * diff)
        with tf.GradientTape() as t:
            t.watch(interp)
            pred = self.discriminate(known, interp)
            grad = t.gradient(pred, [interp])[0]
        slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3, 4]))
        gp = tf.reduce_mean((slopes - 1.) ** 2)
        return gp

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape(persistent=True) as tape:
            ##### Forward pass
            known = stack_known(batch)
            mean, logvar = self.encode(known)
            z = self.reparameterize(mean, logvar)
            sample_logit = self.decode(z)
            sample = tf.nn.sigmoid(sample_logit)
            output = {'predicted_occ': sample, 'predicted_free': 1 - sample}
            metrics = nn.calc_metrics(output, batch)

            #### vae loss
            vae_loss = compute_vae_loss(z, mean, logvar, sample_logit, labels=batch['gt_occ'])
            metrics['loss/vae'] = vae_loss

            ### gan loss
            # fake_occ = tf.cast(sample_logit > 0, tf.float32)
            fake_occ = tf.sigmoid(sample_logit)
            real_pair_est = self.discriminate(known, batch['gt_occ'])
            fake_pair_est = self.discriminate(known, fake_occ)
            gan_loss_g = 10000 * (1 + tf.reduce_mean(-fake_pair_est))
            gan_loss_d_no_gp = 1 + tf.reduce_mean(fake_pair_est - real_pair_est)

            # gradient penalty
            gp = self.gradient_penalty(known, batch['gt_occ'], fake_occ)
            gan_loss_d = gan_loss_d_no_gp + gp

            metrics['loss/gan_g'] = gan_loss_g
            metrics['loss/gan_d'] = gan_loss_d
            metrics['loss/gan_gp'] = gp
            metrics['loss/gan_d_no_gp'] = gan_loss_d_no_gp

            ### apply
            generator_loss = vae_loss + gan_loss_g
            dis_loss = gan_loss_d

        vae_variables = self.encoder.trainable_variables + self.generator.trainable_variables
        vae_gradients = tape.gradient(generator_loss, vae_variables)
        clipped_vae_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in vae_gradients]
        self.optimizer.apply_gradients(list(zip(clipped_vae_gradients, vae_variables)))

        dis_variables = self.discriminator.trainable_variables
        dis_gradients = tape.gradient(dis_loss, dis_variables)
        clipped_dis_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in dis_gradients]
        self.dis_opt.apply_gradients(list(zip(clipped_dis_gradients, dis_variables)))

        m = {k: tf.reduce_mean(metrics[k]) for k in metrics}
        m['loss'] = generator_loss
        return None, m


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


def make_generator(params):
    """Basic VAE decoder"""
    n_features = params['num_latent_layers']
    return tf.keras.Sequential(
        [
            tfl.InputLayer(input_shape=(n_features,)),
            tfl.Dense(4 * 4 * 4 * 512),
            tfl.Activation(tf.nn.relu),
            tfl.Reshape(target_shape=(4, 4, 4, 512)),

            tfl.Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(1, (2, 2, 2), strides=(1, 1, 1), padding="same"),
        ]
    )


def make_discriminator(inp_shape, params):
    """Basic Descriminator"""
    return tf.keras.Sequential(
        [
            tfl.InputLayer(input_shape=inp_shape),

            tfl.Conv3D(16, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.leaky_relu),

            tfl.Conv3D(32, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.leaky_relu),

            tfl.Conv3D(64, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.leaky_relu),

            tfl.Conv3D(128, (2, 2, 2), strides=(2, 2, 2)),
            tfl.Activation(tf.nn.leaky_relu),

            tfl.Flatten(),
            # tfl.Dense(1),
            tfl.Lambda(lambda x: tf.reduce_mean(x, axis=[1])),
            tfl.Activation(tf.nn.sigmoid)
        ]
    )
