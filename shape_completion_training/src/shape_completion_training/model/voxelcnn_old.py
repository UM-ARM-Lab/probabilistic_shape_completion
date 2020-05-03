import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras.layers as tfl
import nn_tools as nn
from nn_tools import MaskedConv3D, p_x_given_y

import IPython


class VoxelCNN(tf.keras.Model):
    def __init__(self, params, batch_size=16):
        super(VoxelCNN, self).__init__()
        self.params = params
        self.layers_dict = {}
        self.layer_names = []
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.001)
        self.setup_model()

    def setup_model(self):
        conv_size = 3

        self.conv_layers = [
            MaskedConv3D(conv_size, 1, 16,      name='masked_conv_1', is_first_layer=True),
            tfl.Activation(tf.nn.elu,           name='conv_1_activation'),
            MaskedConv3D(conv_size, 16, 32,     name='masked_conv_2'),
            tfl.Activation(tf.nn.elu,           name='conv_2_activation'),
            MaskedConv3D(conv_size, 32, 64,     name='masked_conv_3'),
            tfl.Activation(tf.nn.elu,           name='conv_3_activation'),
            # MaskedConv3D(conv_size, 64, 128,  name='masked_conv_4'),
            # tfl.Activation(tf.nn.elu,           name='conv_4_activation'),
            # MaskedConv3D(conv_size, 128, 64,  name='masked_conv_5'),
            # tfl.Activation(tf.nn.elu,           name='conv_5_activation'),
            MaskedConv3D(conv_size, 64, 32,     name='masked_conv_6'),
            tfl.Activation(tf.nn.elu,           name='conv_6_activation'),
            MaskedConv3D(conv_size, 32, 16,     name='masked_conv_7'),
            tfl.Activation(tf.nn.elu,           name='conv_7_activation'),
            MaskedConv3D(conv_size, 16, 1,      name='masked_conv_8'),
            # 
            ]
        if self.params['final_activation'] == 'sigmoid':
            self.conv_layers.append(tfl.Activation(tf.nn.sigmoid,       name='conv_8_activation'))
        elif self.params['final_activation'] == 'elu':
            self.conv_layers.append(tfl.Activation(tf.nn.elu,           name='conv_8_activation'))
        
        # for l in conv_layers:
        #     self._add_layer(l)

    def call(self, inputs, training = False):
        x = inputs['conditioned_occ']

        for l in self.conv_layers:
            x = l(x)
        return {'predicted_occ':x, 'predicted_free':1.0-x}


    @tf.function
    def mse_loss(self, metrics):
        l_occ = tf.reduce_sum(metrics['mse/occ']) * (1.0/self.batch_size)
        return l_occ

    @tf.function
    def train_step(self, batch):
        def reduce(val):
            return tf.reduce_mean(val)
            
        
        def step_fn(batch):
            with tf.GradientTape() as tape:
                output = self(batch, training=True)

                metrics = nn.calc_metrics(output, batch)
                
                if self.params['loss'] == 'cross_entropy':
                    loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(batch['gt_occ'],
                                                                             output['predicted_occ']))
                elif self.params['loss'] == 'mse':
                    loss = self.mse_loss(metrics)
                variables = self.trainable_variables
                gradients = tape.gradient(loss, variables)

                self.opt.apply_gradients(list(zip(gradients, variables)))
                return loss, metrics
            
        loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m



    

class StackedVoxelCNN:
    def __init__(self, params, batch_size):
        self.params = params
        self.model=None
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.Adam(0.001)

        self.make_stack_net(inp_shape = [64,64,64,1])

    def get_model(self):
        return self.model


    def make_stack_net(self, inp_shape):

        model_selector = {
            'v1': lambda: make_stack_net_v1(inp_shape, self.batch_size, self.params),
            'v2': lambda: make_stack_net_v2(inp_shape, self.batch_size, self.params),
            'v3': lambda: make_stack_net_v3(inp_shape, self.batch_size, self.params),
            'v4': lambda: make_stack_net_v4(inp_shape, self.batch_size, self.params),
        }
        self.model = model_selector[self.params['stacknet_version']]()

    def predict(self, elem):
        return self(next(elem.__iter__()))

    def __call__(self, inp):
        model_inp = {k: inp[k] for k in self.model.input.keys()}
        x = self.model(model_inp)
        return x

    @tf.function
    def mse_loss(self, metrics):
        l_occ = tf.reduce_sum(metrics['mse/occ']) * (1.0/self.batch_size)
        return l_occ

    @tf.function
    def train_step(self, batch):
        def reduce(val):
            return tf.reduce_mean(val)
            
        
        def step_fn(batch):
            with tf.GradientTape() as tape:
                output = self(batch)

                metrics = nn.calc_metrics(output, batch)
                
                if self.params['loss'] == 'cross_entropy':
                    cross_ent = tf.keras.losses.binary_crossentropy(batch['gt_occ'],
                                                               output['predicted_occ']))
                    loss = nn.reduce_sum_batch(cross_ent)


                if self.params['stacknet_version'] == 'v4':
                    ae_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(batch['gt_occ'],
                                                                                output['aux_occ']))

                    metrics['loss/aux_loss'] = ae_loss
                    metrics['loss/vcnn_loss'] = loss
                    loss = loss + 0.1*ae_loss
                    
                variables = self.model.trainable_variables
                gradients = tape.gradient(loss, variables)

                clipped_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in gradients]

                self.opt.apply_gradients(list(zip(clipped_gradients, variables)))
                return loss, metrics

        def step_fn_multiloss(batch):
            with tf.GradientTape() as tape:
                output = self(batch)
                metrics = nn.calc_metrics(output, batch)
                loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(batch['gt_occ'],
                                                                         output['predicted_occ']))
                loss = loss / self.batch_size
                metrics['loss/0_step'] = loss
                m = metrics


                # for i in range(1):
                # Multistep part
                if True:
                    i = 0
                    b = {'conditioned_occ': output['predicted_occ']}
                    output = self(b)
                    step_loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(batch['gt_occ'],
                                                                                  output['predicted_occ']))
                    step_loss = step_loss / self.batch_size
                    metrics['loss/{}_step'.format(i+1)] = step_loss

                    loss = tf.cond(metrics['pred|gt/p(predicted_occ|gt_occ)'] > 0.95,
                                   lambda: tf.add(step_loss, loss),
                                   lambda: loss)

                    
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



def make_stack_net_v1(inp_shape, batch_size, params):
    """Basic Stacked VCNN. Like maskedCNN, but moving layers around to remove blind spots"""
    n_filters = 16
    n_per_block = 3
        
    inputs = {'conditioned_occ':tf.keras.Input(batch_size=batch_size, shape=inp_shape)}
    x = inputs['conditioned_occ']

    def bs(x):
        return nn.BackShiftConv3D(n_filters, use_bias=False,
                                  nln=tf.nn.elu)(x)
    def bds(x):
        return nn.BackDownShiftConv3D(n_filters, use_bias=False,
                                      nln=tf.nn.elu)(x)
    def bdrs(x):
        return nn.BackDownRightShiftConv3D(n_filters, use_bias=False,
                                           nln=tf.nn.elu)(x)

    #Front
    f_list = [nn.BackShift()(bs(x))]

    #Upper Front
    uf_list = [nn.BackShift()(bs(x)) + \
               nn.DownShift()(bds(x))]
    
    #Left Upper Front
    luf_list = [nn.BackShift()(bs(x)) + \
                nn.DownShift()(bds(x)) + \
                nn.RightShift()(bdrs(x))]
    
    
    for _ in range(n_per_block):
        f_list.append(bs(f_list[-1]))
        uf_list.append(bds(uf_list[-1]) + f_list[-1])
        luf_list.append(bdrs(luf_list[-1]) + uf_list[-1])
        
        
    x = nn.Conv3D(n_filters=1, filter_size=[1,1,1], use_bias=True)(luf_list[-1])
    
    
    if params['final_activation'] == 'sigmoid':
        x = tf.nn.sigmoid(x)
    elif params['final_activation'] == 'elu':
        x = tf.nn.elu(x)
    elif params['final_activation'] == None:
        pass
    else:
        raise("Unknown param valies for [final activation]: {}".format(params['final_activation']))

    output = {"predicted_occ":x, "predicted_free":1-x}
    return tf.keras.Model(inputs=inputs, output=x)


def make_stack_net_v2(inp_shape, batch_size, params):
    """Stacked VCNN with hourglass shape and unet connections"""
    filter_size = [2,2,2]
    n_filters = [64, 128, 256, 512]

    inputs = {'conditioned_occ':tf.keras.Input(batch_size=batch_size, shape=inp_shape)}
    x = inputs['conditioned_occ']

    # inputs = tf.keras.Input(batch_size=batch_size, shape=inp_shape)
    # x = inputs

    conv_args_strided = {'use_bias': True,
                 # 'filter_size': filter_size,
                 'nln': tf.nn.elu,
                 'strides':[1,2,2,2,1]}
    
    def bs_strided(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)
    def bds_strided(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)
    def bdrs_strided(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    conv_args = {'use_bias': True,
                 # 'filter_size': filter_size,
                 'nln': tf.nn.elu,
                 'strides':[1,1,1,1,1]}
    
    def bs(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)
    def bds(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)
    def bdrs(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    flf = 4 # first_layer_filters

    #Front,     #Upper Front, and     #Left Upper Front
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
        f = tf.concat([tfl.Conv3DTranspose(fs, [2,2,2], strides=[2,2,2])(f), f_list.pop()], axis=4)
        uf = tf.concat([tfl.Conv3DTranspose(fs, [2,2,2], strides=[2,2,2])(uf), uf_list.pop()], axis=4) + f
        luf = tf.concat([tfl.Conv3DTranspose(fs, [2,2,2], strides=[2,2,2])(luf), luf_list.pop()], axis=4) + uf
        
    x = nn.Conv3D(n_filters=1, filter_size=[1,1,1], use_bias=True)(luf)
    
    
    if params['final_activation'] == 'sigmoid':
        x = tf.nn.sigmoid(x)
    elif params['final_activation'] == 'elu':
        x = tf.nn.elu(x)
    elif params['final_activation'] == None or params['final_activation'] == 'None':
        pass
    else:
        raise("Unknown param valies for [final activation]: {}".format(params['final_activation']))

    output = {"predicted_occ":x, "predicted_free":1-x}
    return tf.keras.Model(inputs=inputs, outputs=output)






def make_stack_net_v3(inp_shape, batch_size, params):
    """
    Just an Autoencoder. Used to verify against v4
    """
    filter_size = [2,2,2]
    n_filters = [64, 128, 256, 512]

    inputs = {'conditioned_occ':tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              'known_occ':tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              'known_free':tf.keras.Input(batch_size=batch_size, shape=inp_shape),
    }


    # Autoencoder
    x = tfl.concatenate([inputs['known_occ'], inputs['known_free']], axis=4)

    for n_filter in [64, 128, 256, 512]:
        x = tfl.Conv3D(n_filter, (2,2,2,), use_bias=True, padding="same")(x)
        x = tfl.Activation(tf.nn.relu)(x)
        x = tfl.MaxPool3D((2,2,2))(x)

    x = tfl.Flatten()(x)
    x = tfl.Dense(params['num_latent_layers'], activation='relu')(x)
    x = tfl.Dense(32768, activation='relu')(x)
    x = tfl.Reshape((4,4,4,512))(x)

    for n_filter in [256, 128, 64, 12]:
        x = tfl.Conv3DTranspose(n_filter, (2,2,2,), use_bias=True, strides=2)(x)
        x = tfl.Activation(tf.nn.relu)(x)

    x = tfl.Conv3D(1, (1,1,1), use_bias=True)(x)
    x = tfl.Activation(tf.nn.sigmoid)(x)

    output = {"predicted_occ":x, "predicted_free":1 - x}
    return tf.keras.Model(inputs=inputs, outputs=output)





    
def make_stack_net_v4(inp_shape, batch_size, params):
    """
    Autoencoder combined with VCNN
    """
    inputs = {'conditioned_occ':tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              'known_occ':tf.keras.Input(batch_size=batch_size, shape=inp_shape),
              'known_free':tf.keras.Input(batch_size=batch_size, shape=inp_shape),
    }


    # Autoencoder
    x = tfl.concatenate([inputs['known_occ'], inputs['known_free']], axis=4)

    for n_filter in [64, 128, 256, 512]:
        x = tfl.Conv3D(n_filter, (2,2,2,), use_bias=True, padding="same")(x)
        x = tfl.Activation(tf.nn.relu)(x)
        x = tfl.MaxPool3D((2,2,2))(x)

    x = tfl.Flatten()(x)
    x = tfl.Dense(params['num_latent_layers'], activation='relu')(x)
    x = tfl.Dense(32768, activation='relu')(x)
    x = tfl.Reshape((4,4,4,512))(x)
    auto_encoder_features = x

    for n_filter in [256, 128, 64, 12]:
        x = tfl.Conv3DTranspose(n_filter, (2,2,2,), use_bias=True, strides=2)(x)
        x = tfl.Activation(tf.nn.relu)(x)

    x = tfl.Conv3D(1, (1,1,1), use_bias=True)(x)
    ae_output_before_activation = x
    autoencoder_output = tfl.Activation(tf.nn.sigmoid)(x)




    # VCNN
    filter_size = [2,2,2]
    # n_filters = [64, 128, 256, 512]

    x = inputs['conditioned_occ']
    conv_args_strided = {'use_bias': True,
                         'nln': tf.nn.elu,
                         'strides':[1,2,2,2,1]}
    
    def bs_strided(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)
    def bds_strided(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)
    def bdrs_strided(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args_strided)(x)

    conv_args = {'use_bias': True,
                 'nln': tf.nn.elu,
                 'strides':[1,1,1,1,1]}
    
    def bs(x, n_filters):
        return nn.BackShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)
    def bds(x, n_filters):
        return nn.BackDownShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)
    def bdrs(x, n_filters):
        return nn.BackDownRightShiftConv3D(n_filters, filter_size=filter_size, **conv_args)(x)

    flf = 4 # num_first_layer_filters

    #Front,     #Upper Front, and     #Left Upper Front
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
        f = tf.concat([tfl.Conv3DTranspose(fs, [2,2,2], strides=[2,2,2])(f), f_list.pop()], axis=4)
        f = tfl.Activation(tf.nn.elu)(f)
        uf = tf.concat([tfl.Conv3DTranspose(fs, [2,2,2], strides=[2,2,2])(uf), uf_list.pop(), f], axis=4)
        uf = tfl.Activation(tf.nn.elu)(uf)
        luf = tf.concat([tfl.Conv3DTranspose(fs, [2,2,2], strides=[2,2,2])(luf), luf_list.pop(), uf], axis=4)
        luf = tfl.Activation(tf.nn.elu)(luf)

    x = luf
    
    x = nn.Conv3D(n_filters=1, filter_size=[1,1,1], use_bias=True)(x)
    
    
    if params['final_activation'] == 'sigmoid':
        x = tf.nn.sigmoid(x)
    elif params['final_activation'] == 'elu':
        x = tf.nn.elu(x)
    elif params['final_activation'] == None:
        pass
    else:
        raise("Unknown param valies for [final activation]: {}".format(params['final_activation']))

    


    
    output = {"predicted_occ":x, "predicted_free":1 - x, "aux_occ":autoencoder_output}
    return tf.keras.Model(inputs=inputs, outputs=output)





