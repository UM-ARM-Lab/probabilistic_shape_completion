#! /usr/bin/env python

from model.data_tools import load_data
# from model.network import SimpleNetwork
from model.network import AutoEncoder



if __name__ == "__main__":
    print("hi")
    # load_data()
    # sn = SimpleNetwork()
    sn = AutoEncoder()
    sn.train_and_test(load_data())
