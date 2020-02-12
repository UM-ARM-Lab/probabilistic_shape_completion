#! /usr/bin/env python


import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import data_tools
from model.network import AutoEncoderWrapper
import time
import IPython

shape_map = {"airplane":"02691156",
             "mug":"03797390"}


if __name__ == "__main__":
    cache_fp = join(dirname(__file__), "data/ShapeNetCore.v2_augmented/tfrecords/filepath/ds.cache")

    dataset = data_tools.load_shapenet([shape_map["mug"]])

    batched_ds = dataset.batch(16)
    batched_ds = batched_ds


    i = 0
    t = time.time()
    for elem in batched_ds:
        i+=1
        print(i, elem['gt_occ'].numpy()[0,0,0,0,0], time.time() - t)



