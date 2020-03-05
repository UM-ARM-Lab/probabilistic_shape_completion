#! /usr/bin/env python
import sys
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)
sc_path = join(dirname(abspath(__file__)), "../..")
sys.path.append(sc_path)

from model import data_tools
from model.network import Network

import numpy as np
import IPython


params = {
    'num_latent_layers': 200,
    'translation_pixel_range_x': 10,
    'translation_pixel_range_y': 10,
    'translation_pixel_range_z': 10,
    'is_u_connected': True,
    'final_activation': 'sigmoid',
    'unet_dropout_rate': 0.5,
    'use_final_unet_layer': False,
    'simulate_partial_completion': False,
    'simulate_random_partial_completion': True,
    'network': 'VoxelCNN',
    # 'network': 'AutoEncoder',
}



def compare(net, data):
    e = next(data.batch(1).__iter__())
    completion_all_known = net.model(e)['predicted_occ']
    s = completion_all_known.shape
    completion_one_by_one = np.zeros(s)

    # e['known_occ'][0,:,:,:,0] = 0.0
    # IPython.embed()
    e['known_occ'] = np.zeros(s, dtype=np.float32)
    indices = [(i,j,k) for i in range(s[1]) for j in range(s[2]) for k in range(s[3])]


    mismatch = False
    for ind in indices:
        i = (0, ind[0], ind[1], ind[2], 0)

        # print("Checking: ({}, {}, {})".format(ind[0], ind[1], ind[2]))
        # if e['gt_occ'][i] != 0:
        if True:
            print("Checking: ({}, {}, {})".format(ind[0], ind[1], ind[2]))
            completion_one_by_one = net.model(e)['predicted_occ']
            if completion_one_by_one[i] != completion_all_known[i]:
                print("Completion Mismatch!")
                mismatch = True
                IPython.embed()

            
        e['known_occ'][i] = e['gt_occ'][i]
    if not mismatch:
        print("Everything matches between the completion given GT all at once, and the completion given masked gt")
    



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

    
    net = Network(params, "VoxelCNN_only_mask_first_layer")

    compare(net, data)
    # IPython.embed()

    # sn.train_and_test(data)

