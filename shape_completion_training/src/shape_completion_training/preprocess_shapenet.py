#! /usr/bin/env python

import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import data_tools


if __name__=="__main__":
    data_tools.write_shapenet_to_tfrecord(data_tools.shapenet_labels(["mug"]))
