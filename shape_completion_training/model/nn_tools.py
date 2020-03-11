'''
Utilities used by networks
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras.layers as tfl

import IPython


@tf.function
def p_x_given_y(x, y):
    """
    Returns the reduce p(x|y)
    Clips x from 0 to one, then filters and normalizes by y
    Assumes y is a tensor where every element is 0.0 or 1.0
    """
    clipped = tf.clip_by_value(x, 0.0, 1.0)
    return tf.reduce_sum(clipped * y) / tf.reduce_sum(y)



class MaskedConv3D(tf.keras.layers.Layer):
    def __init__(self, conv_size, in_channels, out_channels, name='masked_conv_3d', is_first_layer=False):
        super(MaskedConv3D, self).__init__(name=name)
        self.conv_size = conv_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_first_layer = is_first_layer

    def build(self, input_shape):
        conv_vars = int(self.conv_size)**3

        if self.is_first_layer:
            n_train = (conv_vars/2) * self.in_channels * self.out_channels
            n_zeros = (conv_vars/2+1) * self.in_channels * self.out_channels
        else:
            n_zeros = (conv_vars/2) * self.in_channels * self.out_channels
            n_train = (conv_vars/2+1) * self.in_channels * self.out_channels

        
        self.a = self.add_weight(name=self.name + 'trainable',
                                 shape=[n_train],
                                 initializer=tf.initializers.GlorotUniform(),
                                 trainable=True)
        self.b = self.add_weight(name=self.name + 'masked',
                                 shape=[n_zeros],
                                 initializer='zeros',
                                 trainable=False)

        self.bias = self.add_weight(name=self.name + 'bias',
                                    shape=[self.out_channels],
                                    initializer=tf.initializers.GlorotUniform(),
                                    trainable=True)

    def call(self, inputs):
        cs = self.conv_size
        f = tf.reshape(tf.concat([self.a,self.b], axis=0),
                       [cs, cs, cs, self.in_channels, self.out_channels])

        x = tf.nn.conv3d(inputs, f, strides=[1,1,1,1,1], padding='SAME')
        return tf.nn.bias_add(x, self.bias)


class Conv3D(tf.keras.layers.Layer):
    def __init__(self, n_filters, filter_size, use_bias,
                 nln=None, name=None):
        super(Conv3D, self).__init__(name=name)
        self.filter_size = filter_size
        self.n_filters = n_filters
        self.b = None
        self.padding='VALID'
        self.use_bias = use_bias
        self.nln = nln

    def build(self, input_shape):
        self.w = self.add_weight(name='weights',
                                 shape=self.filter_size + [input_shape[-1], self.n_filters],
                                 # shape=[3,3,3,1,1],
                                 initializer=tf.initializers.GlorotUniform(),
                                 # initializer=tf.initializers.ones(),
                                 trainable=True)
        if self.use_bias:
            self.b = self.add_weight(name='bias',
                                     shape=[self.n_filters],
                                     initializer=tf.initializers.zeros(),
                                     trainable=True)

    def call(self, x):
        x = tf.nn.conv3d(x, self.w, padding=self.padding, strides=[1,1,1,1,1])
        if self.use_bias:
            x = tf.nn.bias_add(x, self.b)
        if self.nln is not None:
            x = self.nln(x)
        return x
        

class BackShiftConv3D(Conv3D):
    def __init__(self, n_filters, filter_size=[3,3,3], use_bias=True, nln=None):
        super(BackShiftConv3D, self).__init__(n_filters=n_filters,
                                              filter_size=filter_size,
                                              use_bias=use_bias,
                                              nln=nln)

    def call(self, x):
        x = tf.pad(x, [[0,0], [self.filter_size[0]-1, 0],
                       [int((self.filter_size[1]-1)/2), int((self.filter_size[1]-1)/2)],
                       [int((self.filter_size[2]-1)/2), int((self.filter_size[2]-1)/2)],
                       [0,0]])

        return super(BackShiftConv3D, self).call(x)

    
class BackDownShiftConv3D(Conv3D):
    def __init__(self, n_filters, filter_size=[3,3,3], use_bias=True, nln=None):
        super(BackDownShiftConv3D, self).__init__(n_filters=n_filters,
                                                  filter_size=filter_size,
                                                  use_bias=use_bias,
                                                  nln=nln)

    def call(self, x):
        x = tf.pad(x, [[0,0], [self.filter_size[0]-1, 0],
                       [self.filter_size[1]-1, 0],
                       [int((self.filter_size[2]-1)/2), int((self.filter_size[2]-1)/2)],
                       [0,0]])

        return super(BackDownShiftConv3D, self).call(x)

class BackDownRightShiftConv3D(Conv3D):
    def __init__(self, n_filters, filter_size=[3,3,3], use_bias=True, nln=None):
        super(BackDownRightShiftConv3D, self).__init__(n_filters=n_filters,
                                                       filter_size=filter_size,
                                                       use_bias=use_bias,
                                                       nln=nln)

    def call(self, x):
        x = tf.pad(x, [[0,0], [self.filter_size[0]-1, 0],
                       [self.filter_size[1]-1, 0],
                       [self.filter_size[2]-1, 0],
                       [0,0]])

        return super(BackDownRightShiftConv3D, self).call(x)

    

class BackShift(tf.keras.layers.Layer):
    def __init__(self):
        super(BackShift, self).__init__()

    def call(self, x):
        return back_shift(x)
    
class DownShift(tf.keras.layers.Layer):
    def __init__(self):
        super(DownShift, self).__init__()

    def call(self, x):
        return up_shift(x)
    
class RightShift(tf.keras.layers.Layer):
    def __init__(self):
        super(RightShift, self).__init__()

    def call(self, x):
        return right_shift(x)
        


def int_shape(x):
    return list(map(int, x.get_shape()))


def back_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3],xs[4]]), x[:,:xs[1]-1,:,:,:]],1)

def up_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3],xs[4]]), x[:,:,:xs[2]-1,:,:]],2)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],xs[2],1,xs[4]]), x[:,:,:,:xs[3]-1,:]],3)

                    
