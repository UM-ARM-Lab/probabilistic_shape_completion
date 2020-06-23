#! /usr/bin/env python

from shape_completion_training.model import data_tools
from shape_completion_training.model.modelrunner import ModelRunner


params = {
    'num_latent_layers': 200,
    'translation_pixel_range_x': 10,
    'translation_pixel_range_y': 10,
    'translation_pixel_range_z': 10,
    # 'use_final_unet_layer': False,
    'simulate_partial_completion': False,
    'simulate_random_partial_completion': False,
    # 'network': 'VoxelCNN',
    # 'network': 'VAE_GAN',
    'network': 'Augmented_VAE',
    # 'network': 'Conditional_VCNN',
    # 'network': 'NormalizingAE',
    'batch_size': 16,
    'learning_rate': 1e-3,

}


if __name__ == "__main__":
    train_data_shapenet, test_data_shapenet = data_tools.load_shapenet([data_tools.shape_map["mug"]])

    # data = data_ycb
    data = train_data_shapenet

    # if params['network'] == 'VoxelCNN':
    #     sim_input_fn=data_tools.simulate_omniscient_input
    # elif params['network'] == 'AutoEncoder':
    sim_input_fn=data_tools.simulate_2_5D_input
    
    data = data_tools.simulate_input(data,
                                     params['translation_pixel_range_x'],
                                     params['translation_pixel_range_y'],
                                     params['translation_pixel_range_z'],
                                     sim_input_fn=sim_input_fn)
    # data = data_tools.simulate_condition_occ(data,
    #                                          turn_on_prob=params['turn_on_prob'],
    #                                          turn_off_prob=params['turn_off_prob'])

    if params['simulate_partial_completion']:
        data = data_tools.simulate_partial_completion(data)
    if params['simulate_random_partial_completion']:
        data = data_tools.simulate_random_partial_completion(data)

    data = data_tools.add_angle(data)

    mr = ModelRunner(training=True, params=params, group_name=params['network'])
    # mr = ModelRunner(training=True, params=params, group_name=None)

    mr.train_and_test(data)
