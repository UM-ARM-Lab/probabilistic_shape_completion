#! /usr/bin/env python
import shape_completion_training.utils.shapenet_storage
from shape_completion_training.utils import data_tools
from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.model import default_params
import argparse

# params = {
#     'batch_size': 1500,
#     'network': 'RealNVP',
#     'dim': 24,
#     'num_masked': 12,
#     'learning_rate': 1e-5,
#     'translation_pixel_range_x': 10,
#     'translation_pixel_range_y': 10,
#     'translation_pixel_range_z': 10,
# }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args for training")
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--group', default=None)
    args = parser.parse_args()
    params = default_params.get_default_params(group_name=args.group)

    data, _ = data_tools.load_dataset(params['dataset'])


    def _shift(elem):
        return data_tools.shift_bounding_box_only(elem, params['translation_pixel_range_x'],
                                                  params['translation_pixel_range_y'],
                                                  params['translation_pixel_range_z'])

    data = data.map(_shift)


    if args.tmp:
        mr = ModelRunner(training=True, params=params, group_name=None)
    else:
        mr = ModelRunner(training=True, params=params, group_name=args.group)
    # mr = ModelRunner(training=True, params=params, group_name="Flow")
    # IPython.embed()

    mr.train_and_test(data)
