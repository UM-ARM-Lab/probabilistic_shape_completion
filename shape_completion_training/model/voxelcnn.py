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
                acc_occ = tf.math.abs(batch['gt_occ'] - output['predicted_occ'])
                mse_occ = tf.math.square(acc_occ)
                acc_free = tf.math.abs(batch['gt_free'] - output['predicted_free'])
                mse_free = tf.math.square(acc_free)

                unknown_occ = batch['gt_occ'] - batch['known_occ']
                unknown_free = batch['gt_free'] - batch['known_free']
                
                metrics = {"mse/occ": mse_occ, "acc/occ": acc_occ,
                           "mse/free": mse_free, "acc/free": acc_free,
                           "pred|gt/p(predicted_occ|gt_occ)": p_x_given_y(output['predicted_occ'],
                                                                  batch['gt_occ']),
                           "pred|gt/p(predicted_free|gt_free)": p_x_given_y(output['predicted_free'],
                                                                    batch['gt_free']),
                           "pred|known/p(predicted_occ|known_occ)": p_x_given_y(output['predicted_occ'],
                                                                                batch['known_occ']),
                           "pred|known/p(predicted_free|known_free)": p_x_given_y(output['predicted_free'],
                                                                                  batch['known_free']),
                           "pred|gt/p(predicted_occ|gt_free)": p_x_given_y(output['predicted_occ'],
                                                                           batch['gt_free']),
                           "pred|gt/p(predicted_free|gt_occ)": p_x_given_y(output['predicted_free'],
                                                                           batch['gt_occ']),
                           "pred|known/p(predicted_occ|known_free)": p_x_given_y(output['predicted_occ'],
                                                                                 batch['known_free']),
                           "pred|known/p(predicted_free|known_occ)": p_x_given_y(output['predicted_free'],
                                                                                 batch['known_occ']),
                           "pred|unknown/p(predicted_occ|unknown_occ)": p_x_given_y(output['predicted_occ'],
                                                                                    unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_occ)": p_x_given_y(output['predicted_free'],
                                                                                     unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_free)": p_x_given_y(output['predicted_free'],
                                                                                      unknown_free),
                           "pred|unknown/p(predicted_occ|unknown_free)": p_x_given_y(output['predicted_occ'],
                                                                                      unknown_free),
                           "sanity/p(gt_occ|known_occ)": p_x_given_y(batch['gt_occ'], batch['known_occ']),
                           "sanity/p(gt_free|known_occ)": p_x_given_y(batch['gt_free'], batch['known_occ']),
                           "sanity/p(gt_occ|known_free)": p_x_given_y(batch['gt_occ'], batch['known_free']),
                           "sanity/p(gt_free|known_free)": p_x_given_y(batch['gt_free'], batch['known_free']),
                           }
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
        n_filters = 16
        n_per_block = 3
        
        # inputs = tf.keras.Input(shape=inp.get_shape()[1:], batch_size=inp.get_shape()[0])
        inputs = tf.keras.Input(batch_size=self.batch_size, shape=inp_shape)
        # inputs = tf.keras.Input(batch_size=None, shape=inp_shape)
        x = inputs


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


        if self.params['final_activation'] == 'sigmoid':
            x = tf.nn.sigmoid(x)
        elif self.params['final_activation'] == 'elu':
            x = tf.nn.elu(x)
        elif self.params['final_activation'] == None:
            pass
        else:
            raise("Unknown param valies for [final activation]: {}".format(self.params['final_activation']))

        
        
        self.model = tf.keras.Model(inputs=inputs, outputs=x)

    def predict(self, elem):
        # if self.model is None:
        #     IPython.embed()
        #     # self.make_stack_net(next(elem.as_numpy_iterator())['gt_occ'])
        #     self(next(elem.__iter__()))
        return self(next(elem.__iter__()))

    def __call__(self, inp):
        if self.model is None:
            self.make_stack_net(tf.convert_to_tensor(inp['conditioned_occ']))
        x = self.model(inp['conditioned_occ'])
        return {'predicted_occ': x, 'predicted_free':1-x}

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
                acc_occ = tf.math.abs(batch['gt_occ'] - output['predicted_occ'])
                mse_occ = tf.math.square(acc_occ)
                acc_free = tf.math.abs(batch['gt_free'] - output['predicted_free'])
                mse_free = tf.math.square(acc_free)

                unknown_occ = batch['gt_occ'] - batch['known_occ']
                unknown_free = batch['gt_free'] - batch['known_free']
                
                metrics = {"mse/occ": mse_occ, "acc/occ": acc_occ,
                           "mse/free": mse_free, "acc/free": acc_free,
                           "pred|gt/p(predicted_occ|gt_occ)": p_x_given_y(output['predicted_occ'],
                                                                  batch['gt_occ']),
                           "pred|gt/p(predicted_free|gt_free)": p_x_given_y(output['predicted_free'],
                                                                    batch['gt_free']),
                           "pred|known/p(predicted_occ|known_occ)": p_x_given_y(output['predicted_occ'],
                                                                                batch['known_occ']),
                           "pred|known/p(predicted_free|known_free)": p_x_given_y(output['predicted_free'],
                                                                                  batch['known_free']),
                           "pred|gt/p(predicted_occ|gt_free)": p_x_given_y(output['predicted_occ'],
                                                                           batch['gt_free']),
                           "pred|gt/p(predicted_free|gt_occ)": p_x_given_y(output['predicted_free'],
                                                                           batch['gt_occ']),
                           "pred|known/p(predicted_occ|known_free)": p_x_given_y(output['predicted_occ'],
                                                                                 batch['known_free']),
                           "pred|known/p(predicted_free|known_occ)": p_x_given_y(output['predicted_free'],
                                                                                 batch['known_occ']),
                           "pred|unknown/p(predicted_occ|unknown_occ)": p_x_given_y(output['predicted_occ'],
                                                                                    unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_occ)": p_x_given_y(output['predicted_free'],
                                                                                     unknown_occ),
                           "pred|unknown/p(predicted_free|unknown_free)": p_x_given_y(output['predicted_free'],
                                                                                      unknown_free),
                           "pred|unknown/p(predicted_occ|unknown_free)": p_x_given_y(output['predicted_occ'],
                                                                                      unknown_free),
                           "sanity/p(gt_occ|known_occ)": p_x_given_y(batch['gt_occ'], batch['known_occ']),
                           "sanity/p(gt_free|known_occ)": p_x_given_y(batch['gt_free'], batch['known_occ']),
                           "sanity/p(gt_occ|known_free)": p_x_given_y(batch['gt_occ'], batch['known_free']),
                           "sanity/p(gt_free|known_free)": p_x_given_y(batch['gt_free'], batch['known_free']),
                           }
                if self.params['loss'] == 'cross_entropy':
                    loss = tf.reduce_sum(tf.keras.losses.binary_crossentropy(batch['gt_occ'],
                                                                             output['predicted_occ']))
                elif self.params['loss'] == 'mse':
                    loss = self.mse_loss(metrics)
                    
                variables = self.model.trainable_variables
                gradients = tape.gradient(loss, variables)

                self.opt.apply_gradients(list(zip(gradients, variables)))
                return loss, metrics
            
        loss, metrics = step_fn(batch)
        m = {k: reduce(metrics[k]) for k in metrics}
        m['loss'] = loss
        return m

    def summary(self):
        return self.model.summary()
