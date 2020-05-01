#! /usr/bin/env python
import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)
sc_path = join(dirname(abspath(__file__)), "../..")
sys.path.append(sc_path)

from model import data_tools
from model.network import Network
from model.voxelcnn import VoxelCNN, StackedVoxelCNN
import model.nn_tools as nn
import tensorflow.keras.layers as tfl
import tensorflow as tf

import numpy as np
import IPython


params = {
    'num_latent_layers': 200,
    'translation_pixel_range_x': 0,
    'translation_pixel_range_y': 0,
    'translation_pixel_range_z': 0,
    'is_u_connected': True,
    'final_activation': 'sigmoid',
    'unet_dropout_rate': 0.5,
    'use_final_unet_layer': False,
    'simulate_partial_completion': False,
    'simulate_random_partial_completion': True,
    'network': 'VoxelCNN',
    # 'network': 'AutoEncoder',
}



def check_no_backflow(net, data):
    """
    Compares the inference value for each voxel for two inputs:
    -Only previous voxels known
    -All voxels known.
    The structure of the network should mean these give identical results
    """
    e = next(data.batch(1).__iter__())
    completion_all_known = net.model(e)['predicted_occ']
    s = completion_all_known.shape
    completion_one_by_one = np.zeros(s)

    # e['known_occ'][0,:,:,:,0] = 0.0
    # IPython.embed()
    e['conditioned_occ'] = np.zeros(s, dtype=np.float32)
    indices = [(i,j,k) for i in range(s[1]) for j in range(s[2]) for k in range(s[3])]


    mismatch = False
    for ind in indices:
        i = (0, ind[0], ind[1], ind[2], 0)

        if True:
        # if e['gt_occ'][i] == 1.0:
            print("Checking: ({}, {}, {})".format(ind[0], ind[1], ind[2]))
            completion_one_by_one = net.model(e)['predicted_occ']
            if completion_one_by_one[i] != completion_all_known[i]:
                print("Completion Mismatch!")
                mismatch = True
                IPython.embed()
            # IPython.embed()

        if e['conditioned_occ'][i] != e['gt_occ'][i]:
            e['conditioned_occ'][i] = e['gt_occ'][i]
    if not mismatch:
        print("Everything matches between the completion given GT all at once, and the completion given masked gt")


class StackNet(tf.keras.Model):
    def __init__(self):
        super(StackNet, self).__init__()

        self.net_layers = [
            # nn.BackShift(),
            # nn.BackShift(),
            # tfl.Conv3D(1, [3,3,3], kernel_initializer='ones', bias_initializer='zeros',
            #            padding='same')
            # nn.BackShift(),
            # nn.DownShift(),
            # nn.RightShift(),
            # nn.BackShiftConv3D(2),
            # nn.BackShiftConv3D(2),
            nn.BackDownRightShiftConv3D(1)
        ]
        self.back_shift = nn.BackShift()
        self.down_shift = nn.DownShift()
        self.right_shift = nn.RightShift()

        self.b_convs = [
            nn.BackShiftConv3D(1),
            nn.BackShiftConv3D(1)
        ]

        self.bd_convs = [
            nn.BackDownShiftConv3D(1)
            ]

        self.dbr_convs = [
            nn.BackDownShiftConv3D(1)
            ]

    def call(self, x):

        b = self.back_shift(self.b_convs[0](x))
        db = self.back_shift(self.b_convs[1](x)) + self.down_shift(self.bd_convs[0](x))
        
        for l in self.net_layers:
            x = l(x)
        return x

def make_stack_net(inp):
    inputs = tf.keras.Input(shape=inp.get_shape()[1:], batch_size=inp.get_shape()[0])
    x = inputs

    #Front
    f = nn.BackShift()(nn.BackShiftConv3D(1)(x))

    #Upper Front
    uf = nn.BackShift()(nn.BackShiftConv3D(1)(x)) + nn.DownShift()(nn.BackDownShiftConv3D(1)(x))

    #Left Upper Front
    luf = nn.BackShift()(nn.BackShiftConv3D(1)(x)) + \
          nn.DownShift()(nn.BackDownShiftConv3D(1)(x)) + \
          nn.RightShift()(nn.BackDownRightShiftConv3D(1)(x))

    x = luf

    return tf.keras.Model(inputs=inputs, outputs=x)
            

def test_stack_net():
    # inp = np.ones([1,64,64,64,1], dtype=np.float32)
    inp = np.zeros([1,64,64,64,1], dtype=np.float32)
    inp[0,0,0,3,0] = 1.0

    net = StackedVoxelCNN(params={'final_activation':None, 'stacknet_version':'v2'}, batch_size=1)
    e = {'conditioned_occ':tf.convert_to_tensor(inp)}
    out = net(e)['predicted_occ'].numpy()
    print("Input")
    print(inp[0,:,:,:,0])
    print("Output")
    # print(out[0,:,:,:,0])
    print((out[0,0:6,0:7,0:20,0] != 0)+0)
    IPython.embed()



if __name__ == "__main__":
    data_shapenet = data_tools.load_shapenet([data_tools.shape_map["mug"]])

    # data = data_ycb
    data = data_shapenet

    sim_input_fn=data_tools.simulate_omniscient_input
    
    data = data_tools.simulate_input(data,
                                     params['translation_pixel_range_x'],
                                     params['translation_pixel_range_y'],
                                     params['translation_pixel_range_z'],
                                     sim_input_fn=sim_input_fn)
    data = data_tools.simulate_condition_occ(data)


    

    test_stack_net()
     # check_no_backflow(net, data)
    # IPython.embed()

    # sn.train_and_test(data)

    # net = Network(params, "VoxelCNN_only_mask_first_layer")
    # net = Network(params=None, trial_name="VCNN_stacked")
    # net = StackedVoxelCNN(params={'final_activation':None})


