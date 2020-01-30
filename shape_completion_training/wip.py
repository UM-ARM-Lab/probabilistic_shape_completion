#! /usr/bin/env python


import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model.data_tools import load_data
# from model.network import SimpleNetwork
from model.network import AutoEncoder



if __name__ == "__main__":
    print("hi")
    # load_data()
    # sn = SimpleNetwork()
    sn = AutoEncoder()
    sn.train_and_test(load_data())
    # data = load_data()
    # sn.evaluate(data)
    # sn.restore()
    # sn.evaluate(data)
    # sn.evaluate(data)

