import tensorflow as tf
import tensorflow.keras.layers as tfl
import numpy as np

import shape_completion_training.model.nn_tools as nn
from shape_completion_training.model.mykerasmodel import MyKerasModel


def log_normal_pdf(sample, mean, logvar, reduce_axis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=reduce_axis)


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


class ThreeD_rec_gan(MyKerasModel):
    def __init__(self, hparams, batch_size, *args, **kwargs):
        super(ThreeD_rec_gan, self).__init__(hparams, batch_size, *args, **kwargs)
        self.layers_dict = {}
        self.layer_names = []
        self.setup_model()
        self.discriminator = make_discriminator([64, 64, 64, 2], self.hparams)
        self.optimizer = tf.keras.optimizers.Adam(hparams['learning_rate'])
        self.gan_opt = tf.keras.optimizers.Adam(hparams['gan_learning_rate'])

    def get_model(self):
        return self

    def predict(self, elem):
        return self(next(elem.__iter__()))

    def _add_layer(self, layer):
        self.layers_dict[layer.name] = layer
        self.layer_names.append(layer.name)

    def setup_model(self):
        nl = self.hparams['num_latent_layers']
        autoencoder_layers = [
            tfl.Conv3D(64, (2, 2, 2), padding="same", name='conv_4_conv'),
            tfl.Activation(tf.nn.relu, name='conv_4_activation'),
            tfl.MaxPool3D((2, 2, 2), name='conv_4_maxpool'),

            tfl.Conv3D(128, (2, 2, 2), padding="same", name='conv_3_conv'),
            tfl.Activation(tf.nn.relu, name='conv_3_activation'),
            tfl.MaxPool3D((2, 2, 2), name='conv_3_maxpool'),

            tfl.Conv3D(256, (2, 2, 2), padding="same", name='conv_2_conv'),
            tfl.Activation(tf.nn.relu, name='conv_2_activation'),
            tfl.MaxPool3D((2, 2, 2), name='conv_2_maxpool'),

            tfl.Conv3D(512, (2, 2, 2), padding="same", name='conv_1_conv'),
            tfl.Activation(tf.nn.relu, name='conv_1_activation'),
            tfl.MaxPool3D((2, 2, 2), name='conv_1_maxpool'),

            tfl.Flatten(name='flatten'),

            tfl.Dense(nl, activation='relu', name='latent'),

            tfl.Dense(32768, activation='relu', name='expand'),
            tfl.Reshape((4, 4, 4, 512), name='reshape'),

            tfl.Conv3DTranspose(256, (2, 2, 2,), strides=2, name='deconv_1_deconv'),
            tfl.Activation(tf.nn.relu, name='deconv_1_activation'),
            tfl.Conv3DTranspose(128, (2, 2, 2,), strides=2, name='deconv_2_deconv'),
            tfl.Activation(tf.nn.relu, name='deconv_2_activation'),
            tfl.Conv3DTranspose(64, (2, 2, 2,), strides=2, name='deconv_3_deconv'),
            tfl.Activation(tf.nn.relu, name='deconv_3_activation'),

            tfl.Conv3DTranspose(1, (2, 2, 2,), strides=2, name='deconv_4_deconv'),
            # tfl.Activation(tf.nn.relu,                    name='deconv_4_activation'),

            # tfl.Conv3DTranspose(1, (2,2,2,), strides=1,   name='deconv_5_deconv', padding="same"),
        ]

        for l in autoencoder_layers:
            self._add_layer(l)

    def call(self, inputs, training=False):
        known_occ = inputs['known_occ']
        # known_free = inputs['known_free']

        unet = self.hparams['is_u_connected']

        # x = tfl.concatenate([known_occ, known_free], axis=4)
        x = known_occ
        x = self.layers_dict['conv_4_conv'](x)
        x = self.layers_dict['conv_4_activation'](x)
        x = self.layers_dict['conv_4_maxpool'](x)
        u4 = x
        x = self.layers_dict['conv_3_conv'](x)
        x = self.layers_dict['conv_3_activation'](x)
        x = self.layers_dict['conv_3_maxpool'](x)
        u3 = x
        x = self.layers_dict['conv_2_conv'](x)
        x = self.layers_dict['conv_2_activation'](x)
        x = self.layers_dict['conv_2_maxpool'](x)
        u2 = x
        x = self.layers_dict['conv_1_conv'](x)
        x = self.layers_dict['conv_1_activation'](x)
        x = self.layers_dict['conv_1_maxpool'](x)
        u1 = x

        x = self.layers_dict['flatten'](x)
        x = self.layers_dict['latent'](x)
        x = self.layers_dict['expand'](x)
        x = self.layers_dict['reshape'](x)

        if unet:
            x = tfl.concatenate([x, u1], axis=4, name='u_1')
        x = self.layers_dict['deconv_1_deconv'](x)
        x = self.layers_dict['deconv_1_activation'](x)
        if unet:
            x = tfl.concatenate([x, u2], axis=4, name='u_2')
        x = self.layers_dict['deconv_2_deconv'](x)
        x = self.layers_dict['deconv_2_activation'](x)
        if unet:
            x = tfl.concatenate([x, u3], axis=4, name='u_3')
        x = self.layers_dict['deconv_3_deconv'](x)
        x = self.layers_dict['deconv_3_activation'](x)
        if unet:
            x = tfl.concatenate([x, u4], axis=4, name='u_4')
        x = self.layers_dict['deconv_4_deconv'](x)
        # x = self.layers_dict['deconv_4_activation'](x)

        # x = self.layers_dict['deconv_5_deconv'](x)

        # Get logits if training, probabilities if inference
        if not training:
            x = tf.nn.sigmoid(x)
        occ = x
        free = 1 - x
        # occ, free = tf.split(x, 2, axis=4)

        return {'predicted_occ': occ, 'predicted_free': free}

    def discriminate(self, known_input, output):
        inp = tf.concat([known_input, output], axis=4)
        return self.discriminator(inp)

    def gradient_penalty(self, known, real, fake):
        alpha = tf.random.uniform([self.batch_size, 1, 1, 1, 1], 0.0, 1.0)
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
            metrics = {}
            ##### Forward pass
            output = self(batch, training=True)
            output_logits = output['predicted_occ']
            # mean, logvar = self.encode(known)
            # z = self.reparameterize(mean, logvar)
            # sample_logit = self.decode(z)
            # sample = tf.nn.sigmoid(sample_logit)
            # output = {'predicted_occ': sample, 'predicted_free': 1 - sample}
            # metrics = nn.calc_metrics(output, batch)
            #
            # #### vae loss
            # vae_loss = compute_vae_loss(z, mean, logvar, sample_logit, labels=batch['gt_occ'])
            # metrics['loss/vae'] = vae_loss
            aeu_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits,
                                                               labels=batch['gt_occ'])
            aeu_loss = tf.reduce_mean(aeu_loss)
            ### gan loss
            fake_occ = tf.cast(output_logits > 0, tf.float32)
            real_pair_est = self.discriminate(batch['known_occ'], batch['gt_occ'])
            fake_pair_est = self.discriminate(batch['known_occ'], fake_occ)
            gan_loss_g = tf.reduce_mean(-fake_pair_est)
            gan_loss_d_no_gp = tf.reduce_mean(fake_pair_est) - tf.reduce_mean(real_pair_est)

            # gradient penalty
            gp = self.gradient_penalty(batch['known_occ'], batch['gt_occ'], fake_occ)
            gan_loss_d = gan_loss_d_no_gp + 10 * gp

            metrics['loss/gan_g'] = gan_loss_g
            metrics['loss/gan_d'] = gan_loss_d
            metrics['loss/gan_gp'] = gp
            metrics['loss/gan_d_no_gp'] = gan_loss_d_no_gp

            ### apply
            gan_g_w = 20
            aeu_w = 100 - gan_g_w
            generator_loss = aeu_w * aeu_loss + gan_g_w * gan_loss_g
            dis_loss = gan_loss_d

        generator_variables = self.trainable_variables
        generator_gradients = tape.gradient(generator_loss, generator_variables)
        clipped_generator_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in generator_gradients]

        dis_variables = self.discriminator.trainable_variables
        dis_gradients = tape.gradient(dis_loss, dis_variables)
        clipped_dis_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in dis_gradients]

        self.optimizer.apply_gradients(list(zip(clipped_generator_gradients, generator_variables)))
        self.gan_opt.apply_gradients(list(zip(clipped_dis_gradients, dis_variables)))

        m = {k: tf.reduce_mean(metrics[k]) for k in metrics}
        m['loss'] = generator_loss
        return None, m

    @tf.function
    def get_insights(self, variables, gradients):
        final_conv = variables[-1]
        final_grad = gradients[-1]
        insights = {}
        if self.params['is_u_connected'] and self.params['use_final_unet_layer']:
            unet_insights = {
                "weights/know_occ->pred_occ": final_conv[-1][0, 0, 2, 0],
                "weights/know_occ->pred_free": final_conv[-1][0, 0, 2, 1],
                "weights/know_free->pred_occ": final_conv[-1][0, 0, 3, 0],
                "weights/know_free->pred_free": final_conv[-1][0, 0, 3, 1],
                "gradients/know_occ->pred_occ": final_grad[-1][0, 0, 2, 0],
                "gradients/know_occ->pred_free": final_grad[-1][0, 0, 2, 1],
                "gradients/know_free->pred_occ": final_grad[-1][0, 0, 3, 0],
                "gradients/know_free->pred_free": final_grad[-1][0, 0, 3, 1]
            }
            insights.update(unet_insights)
        return insights


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
