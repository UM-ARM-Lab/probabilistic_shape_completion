from colorama import Fore


def get_default_params(group_name=None):
    default_params = {
        'translation_pixel_range_x': 10,
        'translation_pixel_range_y': 10,
        'translation_pixel_range_z': 10,
        'simulate_partial_completion': False,
        'simulate_random_partial_completion': False,
        # 'network': 'VoxelCNN',
        # 'network': 'VAE_GAN',
        # 'network': 'Augmented_VAE',
        # 'network': 'Conditional_VCNN',
        # 'network': 'NormalizingAE',
        'learning_rate': 1e-3,
        'batch_size': 16,
    }

    if group_name is None:
        print(Fore.YELLOW + "Loading default params with no group name" + Fore.RESET)
        return default_params

    group_defaults = {
        "NormalizingAE":
            {
                'num_latent_layers': 200,
                'flow': 'Flow/July_02_10-47-22_d8d84f5d65',
                'network': 'NormalizingAE',
                'use_flow_during_inference': False
            },
        "VAE":
            {
                'num_latent_layers': 200,
                'network': 'VAE'
            },
        "Flow":
            {
                'batch_size': 1500,
                'network': 'RealNVP',
                'dim': 24,
                'num_masked': 12,
                'learning_rate': 1e-5,
                'translation_pixel_range_x': 10,
                'translation_pixel_range_y': 10,
                'translation_pixel_range_z': 10,
            }
    }

    if group_name not in group_defaults:
        print("Group name {} not in group defaults. Add to default_params. "
              "Current groups are {}".format(group_name,
                                             group_defaults.keys()))
        raise Exception("Group name {} not in group defaults.".format(group_name))

    default_params.update(group_defaults[group_name])
    return default_params
