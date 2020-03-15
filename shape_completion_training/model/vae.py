import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras.layers as tfl
import nn_tools as nn
import numpy as np
# from nn_tools import MaskedConv3D, p_x_given_y

import IPython

def stack_known(inp):
    return tf.concat([inp['known_occ'], inp['known_free']], axis=4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


def compute_loss(z, mean, logvar, sample_logit, labels):
    # mean, logvar = model.encode(x)
    # z = model.reparameterize(mean, logvar)
    # x_logit = model.decode(z)
    
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=sample_logit, labels=labels)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)



class VAE(tf.keras.Model):
    def __init__(self, params, batch_size):
        super(VAE, self).__init__()
        self.params = params
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.001)

        self.make_vae(inp_shape = [64,64,64,2])

    def get_model(self):
        return self




    def make_vae(self, inp_shape):
        self.encoder = make_encoder(inp_shape, self.params)
        self.generator = make_generator(self.params)
        
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
        def reduce(val):
            return tf.reduce_mean(val)
            
        
        def step_fn(batch):
            with tf.GradientTape() as tape:
                known = stack_known(batch)
                mean, logvar = self.encode(known)
                z = self.reparameterize(mean, logvar)
                sample_logit = self.decode(z)
                loss = compute_loss(z, mean, logvar, sample_logit, labels=batch['gt_occ'])

                sample = tf.nn.sigmoid(sample_logit)
                output = {'predicted_occ': sample, 'predicted_free': 1 - sample}
                metrics = nn.calc_metrics(output, batch)

                
                    
                variables = self.trainable_variables
                gradients = tape.gradient(loss, variables)

                self.opt.apply_gradients(list(zip(gradients, variables)))
                return loss, metrics

            
        loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m




def make_encoder(inp_shape, params):
    """Basic VAE encoder"""
    n_features = params['num_latent_layers']


    return tf.keras.Sequential(
        [
            tfl.InputLayer(input_shape=inp_shape),

            tfl.Conv3D(64, (2,2,2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2,2,2)),

            tfl.Conv3D(128, (2,2,2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2,2,2)),

            tfl.Conv3D(256, (2,2,2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2,2,2)),

            tfl.Conv3D(512, (2,2,2), padding="same"),
            tfl.Activation(tf.nn.relu),
            tfl.MaxPool3D((2,2,2)),

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
            tfl.Dense(4*4*4*512),
            tfl.Activation(tf.nn.relu),
            tfl.Reshape(target_shape=(4,4,4,512)),

            tfl.Conv3DTranspose(256, (2,2,2), strides=(2,2,2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(128, (2,2,2), strides=(2,2,2)),
            tfl.Activation(tf.nn.relu),
            
            tfl.Conv3DTranspose(64, (2,2,2), strides=(2,2,2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(32, (2,2,2), strides=(2,2,2)),
            tfl.Activation(tf.nn.relu),

            tfl.Conv3DTranspose(1, (2,2,2), strides=(1,1,1), padding="same"),
        ]
    )
