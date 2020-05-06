#! /usr/bin/env python

from __future__ import print_function
import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import data_tools
from shape_completion_training.model.modelrunner import AutoEncoderWrapper
import time
import tensorflow as tf
import progressbar

shape_map = {"airplane":"02691156",
             "mug":"03797390"}


def examine_random_behavior():
    ds = tf.data.Dataset.from_tensor_slices(range(5)).shuffle(2)

    def _rand_map(e):
        return tf.random.uniform(shape=[1], minval=0, maxval=100)

    
    for elem in ds:
        print(elem.numpy())
    print()

    for elem in ds:
        print(elem.numpy())
    print()
    rand_ds = ds.map(_rand_map)
    for elem in rand_ds:
        print(elem.numpy())

    print()
    for elem in rand_ds:
        print(elem.numpy())

    cached_ds = rand_ds.cache('/tmp/tmp.cache')
    print()
    for elem in cached_ds:
        print(elem.numpy())
    print()
    for elem in cached_ds:
        print(elem.numpy())
    
    


def examine_shapenet_loading_behavior():
    dataset = data_tools.load_shapenet([shape_map["mug"]], shuffle=True)
    dataset = data_tools.simulate_input(dataset, 10, 10, 10)
    batched_ds = dataset.batch(16)
    batched_ds = batched_ds



    widgets = [
        ' ', progressbar.Counter(),
        ' [', progressbar.Timer(), '] ',
        ' ', progressbar.Variable("shape"), ' '
        ]


    print()
    with progressbar.ProgressBar(widgets=widgets) as bar:
        for b, elem in enumerate(batched_ds):
            for i in range(16):
                shape_str = "{}:{}".format(elem['shape_category'][i].numpy(),
                                           elem['id'][i].numpy())
                bar.update(b, shape=shape_str)
        # print(i, , time.time() - t)

    print()
    # e = data_tools.shift(elem, 1,2,3)


if __name__ == "__main__":
    # examine_shapenet_loading_behavior()
    examine_random_behavior()
