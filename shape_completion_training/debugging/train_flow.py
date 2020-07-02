#! /usr/bin/env python

from shape_completion_training.model import data_tools
from shape_completion_training.model.modelrunner import ModelRunner

params = {
    'batch_size': 1500,
    'network': 'RealNVP',
    'dim': 24,
    'num_masked': 12,
    'learning_rate': 1e-5,
    'translation_pixel_range_x': 10,
    'translation_pixel_range_y': 10,
    'translation_pixel_range_z': 10,
}

if __name__ == "__main__":
    train_data_shapenet, test_data_shapenet = data_tools.load_shapenet_metadata([data_tools.shape_map["mug"]])

    # data = data_ycb
    data = train_data_shapenet


    def _shift(elem):
        return data_tools.shift_bounding_box_only(elem, params['translation_pixel_range_x'],
                                                  params['translation_pixel_range_y'],
                                                  params['translation_pixel_range_z'])


    data = data.map(_shift)

    mr = ModelRunner(training=True, params=params, group_name="Flow")
    # IPython.embed()

    mr.train_and_test(data)
