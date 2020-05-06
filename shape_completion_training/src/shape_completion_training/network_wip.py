#! /usr/bin/env python

from __future__ import print_function
import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import data_tools
from model.modelrunner import AutoEncoderWrapper
import time
import tensorflow as tf
import progressbar
import IPython


class SimpleNetwork(tf.keras.Model):
    def __init__(self):
        super(SimpleNetwork, self).__init__()

    def call(self, x, training=False):
        return tf.keras.layers.Dropout(rate=0.5)(x, training=training)


if __name__=="__main__":
    print("hi")

    net = SimpleNetwork()
    data = tf.ones([3,5])
    print(net(data, training=True))
