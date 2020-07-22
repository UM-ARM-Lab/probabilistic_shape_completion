#! /usr/bin/env python
import argparse

from shape_completion_training.model import model_evaluator
from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.utils import data_tools
import rospy

MODELS_TO_EVALUATE = [
    # "VAE/VAE_trial_1",
    # "Augmented_VAE/May_21_20-00-00_0000000000",
    # "NormalizingAE/July_02_15-15-06_ede2472d34",
    "VAE_GAN/July_20_23-46-36_8849b5bd57",
    "3D_rec_gan/July_20_19-36-48_7ed486bdf5"
]

if __name__ == "__main__":
    rospy.init_node('evaluation_node')

    parser = argparse.ArgumentParser(description="Process args for training")
    parser.add_argument('--trial', default=None)
    args = parser.parse_args()

    if args.trial is not None:
        MODELS_TO_EVALUATE = [args.trial]



    for trial_path in MODELS_TO_EVALUATE:
        print("Evaluating {}".format(trial_path))
        mr = ModelRunner(training=False, trial_path=trial_path)

        _, test_ds = data_tools.load_dataset(mr.params['dataset'], shuffle=False, metadata_only=True)
        for _ in test_ds_metadata:
            test_set_size += 1
        test_ds = data_tools.load_voxelgrids(test_ds)
        test_ds = data_tools.preprocess_test_dataset(test_ds, mr.params)

        evaluation = {trial_path: model_evaluator.evaluate_model(mr.model, test_ds, test_set_size)}
        model_evaluator.save_evaluation(evaluation)

    print("Finished evaluating dataset")
