#! /usr/bin/env python
from shape_completion_training.model import plausiblility
from shape_completion_training.utils import data_tools
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

        
if __name__ == "__main__":
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    train_dataset, test_dataset = data_tools.load_shapenet_metadata(shuffle=False)
    # dataset = dataset.take(100)

    params = {
    	'apply_slit_occlusion': False,
    	'apply_depth_sensor_noise': False
    }

    fits = plausiblility.compute_icp_fit_dict(test_dataset, params)
    plausiblility.save_plausibilities(fits, "shapenet")

    loaded_fits = plausiblility.load_plausibilities("shapenet")
    print("Finished computing plausibilities")
