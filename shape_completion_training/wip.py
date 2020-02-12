#! /usr/bin/env python


import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import data_tools
from model.network import AutoEncoderWrapper
import IPython

shape_map = {"airplane":"02691156",
             "mug":"03797390"}


if __name__ == "__main__":
    print("hi")
    # data_ycb = load_data(from_record=False)
    data_shapenet = data_tools.load_shapenet([shape_map["mug"]])
    # data_shapenet = data_shapenet.take(1).repeat(400)


    # data = data_ycb
    data = data_shapenet


    params = {'num_latent_layers': 200}
    
    sn = AutoEncoderWrapper(params)
    # IPython.embed()

    sn.train_and_test(data)
    # sn.evaluate(data)
    # sn.restore()
    # sn.evaluate(data)
    # sn.evaluate(data)

