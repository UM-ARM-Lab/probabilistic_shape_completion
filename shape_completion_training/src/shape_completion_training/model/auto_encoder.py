import tensorflow as tf
import tensorflow.keras.layers as tfl

import shape_completion_training.model.nn_tools as nn


class AutoEncoder(tf.keras.Model):
    def __init__(self, params, batch_size=16):
        super(AutoEncoder, self).__init__()
        self.params = params
        self.layers_dict = {}
        self.layer_names = []
        self.setup_model()
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.001)

    def get_model(self):
        return self

    def _add_layer(self, layer):
        self.layers_dict[layer.name] = layer
        self.layer_names.append(layer.name)

    def setup_model(self):

        nl = self.params['num_latent_layers']
        
        autoencoder_layers = [
            tfl.Conv3D(64, (2,2,2), padding="same",  name='conv_4_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_4_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_4_maxpool'),

            tfl.Conv3D(128, (2,2,2), padding="same", name='conv_3_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_3_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_3_maxpool'),

            tfl.Conv3D(256, (2,2,2), padding="same", name='conv_2_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_2_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_2_maxpool'),

            tfl.Conv3D(512, (2,2,2), padding="same", name='conv_1_conv'),
            tfl.Activation(tf.nn.relu,               name='conv_1_activation'),
            tfl.MaxPool3D((2,2,2),                   name='conv_1_maxpool'),

            tfl.Flatten(                             name='flatten'),

            tfl.Dense(nl, activation='relu',         name='latent'),
            
            tfl.Dense(32768, activation='relu',      name='expand'),
            tfl.Reshape((4,4,4,512),                 name='reshape'),
            

            tfl.Conv3DTranspose(256, (2,2,2,), strides=2, name='deconv_1_deconv'),
            tfl.Activation(tf.nn.relu,                    name='deconv_1_activation'),
            tfl.Conv3DTranspose(128, (2,2,2,), strides=2, name='deconv_2_deconv'),
            tfl.Activation(tf.nn.relu,                    name='deconv_2_activation'),
            tfl.Conv3DTranspose(64, (2,2,2,), strides=2,  name='deconv_3_deconv'),
            tfl.Activation(tf.nn.relu,                    name='deconv_3_activation'),
            
            tfl.Conv3DTranspose(1, (2,2,2,), strides=2,   name='deconv_4_deconv'),
            # tfl.Activation(tf.nn.relu,                    name='deconv_4_activation'),
            
            # tfl.Conv3DTranspose(1, (2,2,2,), strides=1,   name='deconv_5_deconv', padding="same"),
        ]
        if self.params['is_u_connected'] and self.params['use_final_unet_layer']:
            extra_unet_layers = [
                tfl.Conv3D(2, (1,1,1,), use_bias=False,                  name='unet_combine'),
                # tfl.Activation(tf.nn.relu,                             name='unet_final_activation'),
            ]
            if self.params['final_activation'] == 'sigmoid':
                extra_unet_layers.append(tfl.Activation(tf.math.sigmoid, name='unet_final_activation'))
            if self.params['final_activation'] == 'relu':
                extra_unet_layers.append(tfl.Activation(tf.nn.relu,      name='unet_final_activation'))

            autoencoder_layers = autoencoder_layers + extra_unet_layers

        for l in autoencoder_layers:
            self._add_layer(l)

    def call(self, inputs, training=False):
        known_occ = inputs['known_occ']
        known_free = inputs['known_free']

        unet = self.params['is_u_connected']

        x = tfl.concatenate([known_occ, known_free], axis=4)

        u5 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u5')(x, training=training)
        x = self.layers_dict['conv_4_conv'](x)
        x = self.layers_dict['conv_4_activation'](x)
        x = self.layers_dict['conv_4_maxpool'](x)
        u4 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u4')(x, training=training)
        x = self.layers_dict['conv_3_conv'](x)
        x = self.layers_dict['conv_3_activation'](x)
        x = self.layers_dict['conv_3_maxpool'](x)
        u3 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u3')(x, training=training)
        x = self.layers_dict['conv_2_conv'](x)
        x = self.layers_dict['conv_2_activation'](x)
        x = self.layers_dict['conv_2_maxpool'](x)
        u2 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u2')(x, training=training)
        x = self.layers_dict['conv_1_conv'](x)
        x = self.layers_dict['conv_1_activation'](x)
        x = self.layers_dict['conv_1_maxpool'](x)
        u1 = tfl.Dropout(rate=self.params['unet_dropout_rate'], name='dropout_u1')(x, training=training)

        x = self.layers_dict['flatten'](x)
        x = self.layers_dict['latent'](x)
        x = self.layers_dict['expand'](x)
        x = self.layers_dict['reshape'](x)

        if (unet):
            x = tfl.concatenate([x, u1], axis=4, name='u_1')
        x = self.layers_dict['deconv_1_deconv'](x)
        x = self.layers_dict['deconv_1_activation'](x)
        if (unet):
            x = tfl.concatenate([x, u2], axis=4, name='u_2')
        x = self.layers_dict['deconv_2_deconv'](x)
        x = self.layers_dict['deconv_2_activation'](x)
        if (unet):
            x = tfl.concatenate([x, u3], axis=4, name='u_3')
        x = self.layers_dict['deconv_3_deconv'](x)
        x = self.layers_dict['deconv_3_activation'](x)
        if (unet):
            x = tfl.concatenate([x, u4], axis=4, name='u_4')
        x = self.layers_dict['deconv_4_deconv'](x)
        # x = self.layers_dict['deconv_4_activation'](x)

        if (unet and self.params['use_final_unet_layer']):
            x = tfl.concatenate([x, u5], axis=4, name='u_5')
            x = self.layers_dict['unet_combine'](x)
            x = self.layers_dict['unet_final_activation'](x)

        # x = self.layers_dict['deconv_5_deconv'](x)

        # Get logits if training, probabilities if inference
        if not training:
            x = tf.nn.sigmoid(x)
        occ = x
        free = 1 - x
        # occ, free = tf.split(x, 2, axis=4)

        return {'predicted_occ': occ, 'predicted_free': free}

    @tf.function
    def mse_loss(self, metrics):
        l_occ = nn.reduce_sum_batch(metrics['mse/occ'])
        l_free = nn.reduce_sum_batch(metrics['mse/free'])
        return l_occ + l_free

    @tf.function
    def train_step(self, batch):
        def reduce(val):
            return tf.reduce_mean(val)

        def step_fn(batch):
            with tf.GradientTape() as tape:
                output_logits = self(batch, training=True)

                x = tf.nn.sigmoid(output_logits['predicted_occ'])

                output = {'predicted_occ': x, 'predicted_free': 1 - x}

                metrics = nn.calc_metrics(output, batch)

                if self.params['loss'] == 'mse':
                    loss = self.mse_loss(metrics)

                elif self.params['loss'] == 'cross_entropy':
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits['predicted_occ'],
                                                                        labels=batch['gt_occ'])
                    loss = nn.reduce_sum_batch(cross_ent)

                variables = self.trainable_variables
                gradients = tape.gradient(loss, variables)

                self.opt.apply_gradients(list(zip(gradients, variables)))
                metrics.update(self.get_insights(variables, gradients))
                return loss, metrics

        loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m

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
