'''
Utilities used by networks
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import tensorflow.keras.layers as tfl
import nn_tools


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

