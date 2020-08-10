#! /usr/bin/env python

import tensorflow as tf

print("-------- GPU ---------")
print("Is there GPU?")
a = tf.test.is_gpu_available()
print(a)
print(tf.test.gpu_device_name())

