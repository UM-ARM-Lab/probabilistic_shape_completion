#! /usr/bin/env python
from shape_completion_training.model import model_evaluator
from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.model import data_tools
import rospy

MODELS_TO_EVALUATE = ["VAE/VAE_trial_1",
    # "Augmented_VAE/May_21_20-00-00_0000000000",
    "NormalizingAE/July_02_15-15-06_ede2472d34"
]

if __name__ == "__main__":
    rospy.init_node('evaluation_node')

    train_ds, test_ds = data_tools.load_shapenet(shuffle=False)
    test_ds = data_tools.simulate_input(test_ds, 0, 0, 0)

    _, test_ds_metadata = data_tools.load_shapenet_metadata(shuffle=False)
    test_set_size = 0
    for _ in test_ds_metadata:
        test_set_size += 1

    # test_ds = test_ds.skip(125).take(1)

    for trial_path in MODELS_TO_EVALUATE:
        print("Evaluating {}".format(trial_path))
        mr = ModelRunner(training=False, trial_path=trial_path)
        evaluation = {trial_path: model_evaluator.evaluate_model(mr.model, test_ds, test_set_size)}
        model_evaluator.save_evaluation(evaluation)

    print("Finished evaluating dataset")
