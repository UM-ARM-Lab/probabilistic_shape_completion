import tensorflow as tf
import tensorflow.keras.layers as tfl

import shape_completion_training.model.nn_tools as nn


class VoxelCNN(tf.keras.Model):
    def __init__(self, params, batch_size):
        super(VoxelCNN, self).__init__()
        self.params = params
        self.model = None
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.00001)
        self.make_stack_net(inp_shape=[64, 64, 64, 1])

    def get_model(self):
        return self.model

    def make_stack_net(self, inp_shape):

        model_selector = {
            # 'v1': lambda: make_stack_net_v1(inp_shape, self.batch_size, self.params),
            'v2': lambda: make_stack_net_v2(inp_shape, self.batch_size, self.params),
            # 'v3': lambda: make_stack_net_v3(inp_shape, self.batch_size, self.params),
            # 'v4': lambda: make_stack_net_v4(inp_shape, self.batch_size, self.params),
        }
        self.model = model_selector[self.params['stacknet_version']]()

    def predict(self, elem):
        return self(next(elem.__iter__()))

    def prep_input(self, inp):
        return {k: inp[k] for k in self.model.input.keys()}

    def call(self, inp, training=False):
        model_inp = self.prep_input(inp)
        x = self.model(model_inp)
        x = tf.nn.sigmoid(x)
        return {'predicted_occ': x, 'predicted_free': 1 - x}

    @tf.function
    def train_step(self, batch):
        def reduce(val):
            return tf.reduce_mean(val)

        def step_fn(batch):
            with tf.GradientTape() as tape:
                output_logits = self.model(self.prep_input(batch))
                sample = tf.nn.sigmoid(output_logits)
                output = {'predicted_occ': sample, 'predicted_free': 1 - sample}
                metrics = nn.calc_metrics(output, batch)

                cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits,
                                                                    labels=batch['gt_occ'])
                loss = nn.reduce_sum_batch(cross_ent)
                variables = self.model.trainable_variables
                gradients = tape.gradient(loss, variables)
                clipped_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in gradients]
                self.opt.apply_gradients(list(zip(clipped_gradients, variables)))
                return loss, metrics

        def step_fn_multiloss(batch):
            with tf.GradientTape() as tape:
                output_logits = self.model(self.prep_input(batch))
                sample = tf.nn.sigmoid(output_logits)
                output = {'predicted_occ': sample, 'predicted_free': 1 - sample}
                metrics = nn.calc_metrics(output, batch)

                cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits,
                                                                    labels=batch['gt_occ'])
                loss = nn.reduce_sum_batch(cross_ent)

                metrics['loss/0_step'] = loss
                m = metrics

                for i in range(5):
                    # Multistep part
                    # if True:
                    # i = 0
                    b = {'conditioned_occ': tf.cast(output_logits > 0, tf.float32)}
                    output_logits = self.model(b)
                    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=output_logits,
                                                                        labels=batch['gt_occ'])
                    step_loss = nn.reduce_sum_batch(cross_ent)
                    metrics['loss/{}_step'.format(i + 1)] = step_loss

                    # loss = loss + step_loss

                    # loss = tf.cond(metrics['pred|gt/p(predicted_occ|gt_occ)'] > 0.95,
                    #                lambda: tf.add(step_loss, loss),
                    #                lambda: loss)

                variables = self.model.trainable_variables
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

    def summary(self):
        return self.model.summary()


def make_stack_net_v2(inp_shape, batch_size, params):
    """Stacked VCNN with hourglass shape and unet connections"""
    filter_size = [2, 2, 2]
    n_filters = [64, 128, 256, 512]

    inputs = {'conditioned_occ': tf.keras.Input(batch_size=batch_size, shape=inp_shape)}
    x = inputs['conditioned_occ']

    # inputs = tf.keras.Input(batch_size=batch_size, shape=inp_shape)
    # x = inputs

    conv_args_strided = {'use_bias': True,
                         # 'filter_size': filter_size,
                         'nln': tf.nn.elu,
                         'strides': [1, 2, 2, 2, 1]}

    def bs_strided(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    def bds_strided(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    def bdrs_strided(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    conv_args = {'use_bias': True,
                 # 'filter_size': filter_size,
                 'nln': tf.nn.elu,
                 'strides': [1, 1, 1, 1, 1]}

    def bs(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    def bds(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    def bdrs(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    flf = 4  # first_layer_filters

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

    for fs in n_filters:
        f_list.append(bs_strided(f_list[-1], fs))
        uf_list.append(bds_strided(uf_list[-1], fs) + f_list[-1])
        luf_list.append(bdrs_strided(luf_list[-1], fs) + uf_list[-1])

    f = f_list.pop()
    uf = uf_list.pop()
    luf = luf_list.pop()

    for fs in reversed(n_filters):
        f = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(f), f_list.pop()], axis=4)
        uf = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(uf), uf_list.pop()], axis=4) + f
        luf = tf.concat([tfl.Conv3DTranspose(fs, [2, 2, 2], strides=[2, 2, 2])(luf), luf_list.pop()], axis=4) + uf

    x = nn.Conv3D(n_filters=1, filter_size=[1, 1, 1], use_bias=True)(luf)

    if params['final_activation'] == 'sigmoid':
        x = tf.nn.sigmoid(x)
    elif params['final_activation'] == 'elu':
        x = tf.nn.elu(x)
    elif params['final_activation'] == None or params['final_activation'] == 'None':
        pass
    else:
        raise ("Unknown param valies for [final activation]: {}".format(params['final_activation']))

    output = x
    return tf.keras.Model(inputs=inputs, outputs=output)
