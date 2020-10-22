#! /usr/bin/env python
import argparse

from shape_completion_training.plausible_diversity import model_evaluator
from shape_completion_training.model.model_runner import ModelRunner
from shape_completion_training.utils import data_tools
import rospy

# Submission models
# "VAE/VAE_trial_1",
# "Augmented_VAE/May_21_20-00-00_0000000000",
# "VAE/July_07_12-09-24_7f65111254",
# "PSSNet/July_02_15-15-06_ede2472d34",
# "VAE_GAN/July_20_23-46-36_8849b5bd57",
# "3D_rec_gan/July_20_19-36-48_7ed486bdf5"
# "PSSNet_YCB/July_24_11-21-46_f2aea4d768",
# "VAE_YCB/July_24_11-21-49_f2aea4d768",
# "VAE_GAN_YCB/July_25_22-50-44_0f55a0f6b3",
# "3D_rec_gan_YCB/July_25_22-51-08_0f55a0f6b3"

MODELS_TO_EVALUATE = [
    "VAE/September_12_15-46-26_f87bdf38d4",
    "PSSNet/September_10_21-15-32_f87bdf38d4",
    "VAE_GAN/September_12_15-08-29_f87bdf38d4",
    "3D_rec_gan/September_12_15-47-07_f87bdf38d4"
    # "PSSNet_YCB/July_24_11-21-46_f2aea4d768",
    # "VAE_YCB/July_24_11-21-49_f2aea4d768",
    # "VAE_GAN_YCB/July_25_22-50-44_0f55a0f6b3",
    # "3D_rec_gan_YCB/July_25_22-51-08_0f55a0f6b3"
]

# 30 slit params
slit_params = {
    "slit_start": 17,
    "slit_width": 30
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process args for training")
    parser.add_argument('--trial', default=None)
    parser.add_argument('--num_network_samples', default=100, type=int)
    args = parser.parse_args()

    if args.trial is not None:
        MODELS_TO_EVALUATE = [args.trial]

    rospy.init_node('evaluation_node_' + ("_".join(MODELS_TO_EVALUATE).replace("/", "").replace("-", "_")).lower())

    for trial_path in MODELS_TO_EVALUATE:
        print("Evaluating {}".format(trial_path))
        mr = ModelRunner(training=False, trial_path=trial_path)
        if mr.params['dataset'] == 'ycb':
            mr.params.update(slit_params)

        _, test_ds = data_tools.load_dataset(mr.params['dataset'], shuffle=False, metadata_only=True)
        test_set_size = 0
        for _ in test_ds:
            test_set_size += 1
        test_ds = data_tools.load_voxelgrids(test_ds)
        test_ds = data_tools.preprocess_test_dataset(test_ds, mr.params)

        evaluation = {trial_path: model_evaluator.evaluate_model(mr.model, test_ds, test_set_size,
                                                                 mr.params['dataset'],
                                                                 num_particles=args.num_network_samples)}
        model_evaluator.save_evaluation(evaluation)

    print("Finished evaluating dataset")
