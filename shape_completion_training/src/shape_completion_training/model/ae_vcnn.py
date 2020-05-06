import tensorflow as tf
import tensorflow.keras.layers as tfl

import shape_completion_training.model.nn_tools as nn


class AE_VCNN(tf.keras.Model):
    """
    Autoencoder combined with VCNN by running the AE and VCNN separately, then combining the outputs
    """

    def __init__(self, params, batch_size):
        super(AE_VCNN, self).__init__()
        self.params = params
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.0001)
        self.make_stack_net(inp_shape=[64, 64, 64, 1])

    def get_model(self):
        return self

    def make_stack_net(self, inp_shape):
        # self.model = make_stack_net_v4(inp_shape, self.batch_size, self.params)
        self.encoder = make_encoder(inp_shape, self.batch_size, self.params)
        self.decoder = make_decoder(inp_shape, self.batch_size, self.params)
        self.cvcnn = make_cvcnn(inp_shape, self.batch_size, self.params)

    def predict(self, elem):
        return self(next(elem.__iter__()))

    def prep_ae_input(self, inp):
        return {k: inp[k] for k in self.encoder.input.keys()}

    def prep_cvcnn_inputs(self, conditioned_occ, ae_logits):
        return {'conditioned_occ': conditioned_occ, 'ae_logits': ae_logits}

    def call(self, inp, training=False):
        ae_features = self.encoder(self.prep_ae_input(inp))
        ae_logits = self.decoder(ae_features)
        x = self.cvcnn(self.prep_cvcnn_inputs(inp['conditioned_occ'], ae_logits))
        p_occ = tf.nn.sigmoid(x['p_occ_logits'])
        ae_occ = tf.nn.sigmoid(ae_logits)
        return {'predicted_occ': p_occ, 'predicted_free': 1 - p_occ, 'aux_occ': ae_occ}

    @tf.function
    def train_step(self, batch):
        def reduce(val):
            return tf.reduce_mean(val)

        def step_fn(batch):
            with tf.GradientTape() as tape:
                ae_features = self.encoder(self.prep_ae_input(batch))
                ae_logits = self.decoder(ae_features)

                cvcnn = self.cvcnn(self.prep_cvcnn_inputs(batch['conditioned_occ'], ae_logits))

                p_occ = tf.nn.sigmoid(cvcnn['p_occ_logits'])
                output = {'predicted_occ': p_occ, 'predicted_free': 1 - p_occ}
                metrics = nn.calc_metrics(output, batch)

                cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=cvcnn['p_occ_logits'],
                                                                    labels=batch['gt_occ'])
                vcnn_loss = nn.reduce_sum_batch(cross_ent)

                ae_loss = nn.reduce_sum_batch(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=ae_logits,
                                                            labels=batch['gt_occ']))
                metrics['loss/aux_loss'] = ae_loss

                loss = vcnn_loss + 10 * ae_loss

                variables = self.trainable_variables
                gradients = tape.gradient(loss, variables)
                clipped_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in gradients]
                self.opt.apply_gradients(list(zip(clipped_gradients, variables)))
                return loss, metrics

        def step_fn_multiloss(batch):
            raise Exception("Not Implemented Yet")
            with tf.GradientTape() as tape:
                ae_features = self.encoder(self.prep_ae_input(batch))
                ae_output = self.decoder(ae_features)
                ae_loss = nn.reduce_sum_batch(
                    tf.nn.sigmoid_cross_entropy_with_logits(logits=ae_output,
                                                            labels=batch['gt_occ']))

                cvcnn_inp = self.prep_cvcnn_inputs(batch['conditioned_occ'], ae_features)
                loss = ae_loss

                for i in range(6):
                    cvcnn = self.cvcnn(cvcnn_inp)

                    if i == 0:
                        p_occ = tf.nn.sigmoid(cvcnn['p_occ_logits'])
                        output = {'predicted_occ': p_occ, 'predicted_free': 1 - p_occ}
                        metrics = nn.calc_metrics(output, batch)

                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=cvcnn['p_occ_logits'],
                                                                        labels=batch['gt_occ'])
                    step_loss = nn.reduce_sum_batch(cross_ent)
                    loss = loss + step_loss
                    metrics['loss/{}_step'.format(i)] = step_loss

                    cvcnn_inp = self.prep_cvcnn_inputs(tf.cast(cvcnn['p_occ_logits'] > 0, tf.float32),
                                                       ae_features)

                metrics['loss/aux_loss'] = ae_loss

                variables = self.trainable_variables
                gradients = tape.gradient(loss, variables)
                clipped_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in gradients]
                self.opt.apply_gradients(list(zip(clipped_gradients, variables)))
                return loss, metrics

        if self.params['multistep_loss']:
            loss, metrics = step_fn_multiloss(batch)
        else:
            loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m


def make_encoder(inp_shape, batch_size, params):
    """Encoder of the autoencoder"""
    inputs = inputs = {'conditioned_occ': tf.keras.Input(batch_size=batch_size, shape=inp_shape),
                       'known_occ': tf.keras.Input(batch_size=batch_size, shape=inp_shape),
                       'known_free': tf.keras.Input(batch_size=batch_size, shape=inp_shape),
                       }
    # Autoencoder
    x = tfl.concatenate([inputs['known_occ'], inputs['known_free']], axis=4)

    for n_filter in [64, 128, 256, 512]:
        x = tfl.Conv3D(n_filter, (2, 2, 2,), use_bias=True, padding="same")(x)
        x = tfl.Activation(tf.nn.relu)(x)
        x = tfl.MaxPool3D((2, 2, 2))(x)

    x = tfl.Flatten()(x)
    x = tfl.Dense(params['num_latent_layers'], activation='relu')(x)
    x = tfl.Dense(32768, activation='relu')(x)
    x = tfl.Reshape((4, 4, 4, 512))(x)
    auto_encoder_features = x
    return tf.keras.Model(inputs=inputs, outputs=auto_encoder_features)


def make_decoder(inp_shape, batch_size, params):
    """Decoder of the autoencder"""
    inputs = tf.keras.Input(batch_size=batch_size, shape=(4, 4, 4, 512))

    x = inputs

    for n_filter in [256, 128, 64, 12]:
        x = tfl.Conv3DTranspose(n_filter, (2, 2, 2,), use_bias=True, strides=2)(x)
        x = tfl.Activation(tf.nn.relu)(x)

    x = tfl.Conv3D(1, (1, 1, 1), use_bias=True)(x)
    ae_output_logits = x

    return tf.keras.Model(inputs=inputs, outputs=ae_output_logits)


def make_cvcnn(inp_shape, batch_size, params):
    inputs = {'conditioned_occ': tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              'ae_logits': tf.keras.Input(batch_size=batch_size, shape=inp_shape)
              }

    ae_logits = inputs['ae_logits']

    # VCNN
    filter_size = [2, 2, 2]
    # n_filters = [64, 128, 256, 512]

    x = inputs['conditioned_occ']
    conv_args_strided = {'use_bias': True,
                         'nln': tf.nn.elu,
                         'strides': [1, 2, 2, 2, 1]}

    def bs_strided(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    def bds_strided(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    def bdrs_strided(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    conv_args = {'use_bias': True,
                 'nln': tf.nn.elu,
                 'strides': [1, 1, 1, 1, 1]}

    def bs(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    def bds(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    def bdrs(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    flf = 4  # num_first_layer_filters

    # Front,     #Upper Front, and     #Left Upper Front
    f_1 = nn.BackShift()(bs(x, flf))
    uf_1 = nn.BackShift()(bs(x, flf)) + \
           nn.DownShift()(bds(x, flf))
    luf_1 = nn.BackShift()(bs(x, flf)) + \
            nn.DownShift()(bds(x, flf)) + \
            nn.RightShift()(bdrs(x, flf))

    for i in range(2):
        f_1 = bs(f_1, flf)
        uf_1 = bds(uf_1, flf) + f_1
        luf_1 = bdrs(luf_1, flf) + uf_1

    f_list = [f_1]
    uf_list = [uf_1]
    luf_list = [luf_1]

    for fs in [64, 128, 256, 512]:
        f_list.append(bs_strided(f_list[-1], fs))
        uf_list.append(bds_strided(uf_list[-1], fs) + f_list[-1])
        luf_list.append(bdrs_strided(luf_list[-1], fs) + uf_list[-1])

    f = f_list.pop()
    uf = uf_list.pop()
    luf = luf_list.pop()

    for fs in [256, 128, 64, 4]:
        f = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(f), f_list.pop()], axis=4)
        f = tfl.Activation(tf.nn.elu)(f)
        uf = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(uf), uf_list.pop(), f], axis=4)
        uf = tfl.Activation(tf.nn.elu)(uf)
        luf = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(luf), luf_list.pop(), uf], axis=4)
        luf = tfl.Activation(tf.nn.elu)(luf)

    x = luf
    x = tf.concat([x, ae_logits], axis=4)

    x = nn.Conv3D(n_filters=1, filter_size=[1, 1, 1], use_bias=True)(x)

    output = {"p_occ_logits": x}
    return tf.keras.Model(inputs=inputs, outputs=output)


def make_stack_net_v4(inp_shape, batch_size, params):
    """
    Autoencoder combined with VCNN
    """
    inputs = {'conditioned_occ': tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              'known_occ': tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              'known_free': tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              }

    # Autoencoder
    x = tfl.concatenate([inputs['known_occ'], inputs['known_free']], axis=4)

    for n_filter in [64, 128, 256, 512]:
        x = tfl.Conv3D(n_filter, (2, 2, 2,), use_bias=True, padding="same")(x)
        x = tfl.Activation(tf.nn.relu)(x)
        x = tfl.MaxPool3D((2, 2, 2))(x)

    x = tfl.Flatten()(x)
    x = tfl.Dense(params['num_latent_layers'], activation='relu')(x)
    x = tfl.Dense(32768, activation='relu')(x)
    x = tfl.Reshape((4, 4, 4, 512))(x)
    auto_encoder_features = x

    for n_filter in [256, 128, 64, 12]:
        x = tfl.Conv3DTranspose(n_filter, (2, 2, 2,), use_bias=True, strides=2)(x)
        x = tfl.Activation(tf.nn.relu)(x)

    x = tfl.Conv3D(1, (1, 1, 1), use_bias=True)(x)
    ae_output_before_activation = x
    # autoencoder_output = tfl.Activation(tf.nn.sigmoid)(x)

    # VCNN
    filter_size = [2, 2, 2]
    # n_filters = [64, 128, 256, 512]

    x = inputs['conditioned_occ']
    conv_args_strided = {'use_bias': True,
                         'nln': tf.nn.elu,
                         'strides': [1, 2, 2, 2, 1]}

    def bs_strided(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    def bds_strided(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    def bdrs_strided(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    conv_args = {'use_bias': True,
                 'nln': tf.nn.elu,
                 'strides': [1, 1, 1, 1, 1]}

    def bs(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    def bds(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    def bdrs(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    flf = 4  # num_first_layer_filters

    # Front,     #Upper Front, and     #Left Upper Front
    f_1 = nn.BackShift()(bs(x, flf))
    uf_1 = nn.BackShift()(bs(x, flf)) + \
           nn.DownShift()(bds(x, flf))
    luf_1 = nn.BackShift()(bs(x, flf)) + \
            nn.DownShift()(bds(x, flf)) + \
            nn.RightShift()(bdrs(x, flf))

    for i in range(2):
        f_1 = bs(f_1, flf)
        uf_1 = bds(uf_1, flf) + f_1
        luf_1 = bdrs(luf_1, flf) + uf_1

    f_list = [f_1]
    uf_list = [uf_1]
    luf_list = [luf_1]

    for fs in [64, 128, 256, 512]:
        f_list.append(bs_strided(f_list[-1], fs))
        uf_list.append(bds_strided(uf_list[-1], fs) + f_list[-1])
        luf_list.append(bdrs_strided(luf_list[-1], fs) + uf_list[-1])

    f = f_list.pop()
    uf = uf_list.pop()
    luf = tf.concat([luf_list.pop(), auto_encoder_features], axis=4)

    for fs in [256, 128, 64, 4]:
        f = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(f), f_list.pop()], axis=4)
        f = tfl.Activation(tf.nn.elu)(f)
        uf = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(uf), uf_list.pop(), f], axis=4)
        uf = tfl.Activation(tf.nn.elu)(uf)
        luf = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(luf), luf_list.pop(), uf], axis=4)
        luf = tfl.Activation(tf.nn.elu)(luf)

    x = luf

    x = nn.Conv3D(n_filters=1, filter_size=[1, 1, 1], use_bias=True)(x)

    output = {"p_occ_logits": x, "aux_logits": ae_output_before_activation}
    return tf.keras.Model(inputs=inputs, outputs=output)
