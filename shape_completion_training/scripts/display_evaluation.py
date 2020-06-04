#! /usr/bin/env python
import rospy
from shape_completion_training.model import model_evaluator
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher
from shape_completion_training.model import data_tools
from shape_completion_training.model import utils
from shape_completion_training.model import plausiblility
from shape_completion_training.model.modelrunner import ModelRunner
import tensorflow as tf
import numpy as np
from shape_completion_training.voxelgrid import conversions
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# data_names = ["model", "shape", "best_sample_gt", ""


def display_histogram(evaluation):
    data = {name: [] for name in ["model", "shape", "closest_sample_to_plausible"]}
    for model_name, model_evaluation in evaluation.items():
        print("Processing data for {}".format(model_name))
        for shape_name, shape_evaluation in model_evaluation.items():
            angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
            if not 250 < angle < 290:
                continue
            d = shape_evaluation['particle_distances']
            for closest_particle in list(np.min(d, axis=1)):
                data["model"].append(model_name)
                data["shape"].append(shape_name + model_name)
                data["closest_sample_to_plausible"].append(closest_particle)
            # fmri = sns.load_dataset("fmri")
            # sns.lineplot(x="timepoint", y="signal", hue="region", style="event", data=fmri)
    df = pd.DataFrame(data, columns=data.keys())
    sns.set(style="darkgrid")
    sns.lineplot(x="shape", y="closest_sample_to_plausible", data=df, hue="model")
    plt.show()
    print("wait")


def display_voxelgrids(evaluation):
    # model_name = evaluation.keys()[1]
    model_name = "Augmented_VAE/May_21_20-00-00_0000000000"
    print("Showing results for {}".format(model_name))
    model = ModelRunner(training=False, trial_path=model_name).model
    # shape_name = evaluation[model_name].keys()[0]
    # shape_evaluation = evaluation[model_name][shape_name]
    for shape_name, shape_evaluation in evaluation[model_name].items():
        angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
        if not 250 < angle < 290:
            continue
        # display_evaluation_for_shape(model, shape_name, shape_evaluation)
        plausibles = model_evaluator.get_plausibles(shape_name)

        elem = sn.get(shape_name)
        # e = {k: tf.expand_dims(v, axis=0) for k, v in elem.items()}
        print("Sampling particles for {}".format(shape_name))
        particles = model_evaluator.sample_particles(model, utils.add_batch_to_dict(elem),
                                                     num_particles=len(shape_evaluation['particle_distances'][0]))
        VG_PUB.publish_elem(elem)

        best_particles = np.argmin(shape_evaluation['particle_distances'], axis=1)
        for i, plausible in enumerate(plausibles):
            VG_PUB.publish("plausible", plausible)
            VG_PUB.publish("predicted_occ", particles[best_particles[i]])
            print("Distance is {}".format(np.min(shape_evaluation['particle_distances'], axis=1)[i]))
            rospy.sleep(1)

def display_evaluation_for_shape(model, shape_name, shape_evaluation):
    plausibles = model_evaluator.get_plausibles(shape_name)

    elem = sn.get(shape_name)
    # e = {k: tf.expand_dims(v, axis=0) for k, v in elem.items()}
    print("Sampling particles for {}".format(shape_name))
    particles = model_evaluator.sample_particles(model, utils.add_batch_to_dict(elem),
                                                 num_particles=len(shape_evaluation['particle_distances'][0]))
    VG_PUB.publish_elem(elem)

    best_particles = np.argmin(shape_evaluation['particle_distances'], axis=1)
    for i, plausible in enumerate(plausibles):
        VG_PUB.publish("plausible", plausible)
        VG_PUB.publish("predicted_occ", particles[best_particles[i]])
        rospy.sleep(1)


def display_coverage(particles, plausibles):
    pass


if __name__ == "__main__":
    rospy.init_node('evaluation_display')
    VG_PUB = VoxelgridPublisher()

    full_evaluation = model_evaluator.load_evaluation()
    print("Loading addressable shapenet")
    sn = data_tools.get_addressible_shapenet(use_train=False)
    display_voxelgrids(full_evaluation)
    # display_histogram(full_evaluation)
