from shape_completion_training.model import model_evaluator
from shape_completion_training.model.modelrunner import ModelRunner
from shape_completion_training.model import data_tools

MODELS_TO_EVALUATE = ["Augmented_VAE/May_21_20-00-00_0000000000"]

if __name__ == "__main__":
    train_ds, test_ds = data_tools.load_shapenet(shuffle=False)
    test_ds = data_tools.simulate_input(test_ds, 0, 0, 0)

    _, test_ds_metadata = data_tools.load_shapenet_metadata(shuffle=False)
    test_set_size = 0
    for _ in test_ds_metadata:
        test_set_size += 1

    for model_name in MODELS_TO_EVALUATE:
        mr = ModelRunner(training=False, trial_path=model_name)
        evaluation = model_evaluator.evaluate_model(mr.model, test_ds, test_set_size)
        model_evaluator.save_evaluation(evaluation)
        print("Finished evaluating dataset")
