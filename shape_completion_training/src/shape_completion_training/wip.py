#! /usr/bin/env python
from __future__ import print_function

from shape_completion_training.model.modelrunner import ModelRunner, VoxelCNN
import tensorflow as tf

shape_map = {"airplane":"02691156",
             "mug":"03797390"}


if __name__ == "__main__":
    print("Deprecated. Use `train.py`")


    

    


    initializer = tf.initializers.GlorotUniform()
    a = tf.Variable(initializer([13*4]), trainable=True)
    b = tf.Variable(tf.zeros([14*4]), trainable=False)

    f = tf.reshape(tf.concat([a,b], axis=0), [3,3,3,2, 2])

    vcnn = VoxelCNN(0)
    t = vcnn.get_masked_conv_tensor(3,2,2)

    IPython.embed()


    
    
    # data_shapenet = data_tools.load_shapenet([shape_map["mug"]])



    # # data = data_ycb
    # data = data_shapenet
    # data = data_tools.simulate_input(data, 10, 10, 10)


    # elem = next(data.__iter__())
    # gt = elem['gt_occ']
    
    # # data_tools.simulate_first_random_input(gt)
    # data_tools.simulate_random_partial_completion_input(gt)


    
    # sn = AutoEncoderWrapper()
    # IPython.embed()

    
    # sn.train_and_test(data)
    # # sn.evaluate(data)
    # # sn.restore()
    # # sn.evaluate(data)
    # # sn.evaluate(data)

