import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras.layers as tfl
import data_tools
import filepath_tools
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
                    loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(batch['gt_occ'],
                                                                             output['predicted_occ']))
                    loss = loss / self.batch_size
                elif self.params['loss'] == 'mse':
                    loss = self.mse_loss(metrics) / self.batch_size
                    
                variables = self.model.trainable_variables
                gradients = tape.gradient(loss, variables)

                clipped_gradients = [tf.clip_by_value(g, -1e6, 1e6) for g in gradients]

                self.opt.apply_gradients(list(zip(clipped_gradients, variables)))
                return loss, metrics
            
        loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m

    def summary(self):
        return self.model.summary()



def make_stack_net_v1(inp_shape, batch_size, params):
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
    elif params['final_activation'] == None:
        pass
    else:
        raise("Unknown param valies for [final activation]: {}".format(params['final_activation']))

    return tf.keras.Model(inputs=inputs, outputs=x)
