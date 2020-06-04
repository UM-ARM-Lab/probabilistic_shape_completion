#! /usr/bin/env python
import rospy
from shape_completion_training.model import model_evaluator
from bsaund_shape_completion.voxelgrid_publisher import VoxelgridPublisher
from shape_completion_training.model import data_tools
from shape_completion_training.model import plausiblility
from shape_completion_training.model.modelrunner import ModelRunner
import tensorflow as tf
import numpy as np
from shape_completion_training.voxelgrid import conversions


def display_histogram(evaluation):
    for model_name, model_evaluation in evaluation.items():
        print("Displaying evaluation for {}".format(model_name))
        for shape_name, shape_evaluation in model_evaluation.items():
            d = shape_evaluation['particle_distances']
            print("wait")


def display_voxelgrids(evaluation):
    model_name = evaluation.keys()[0]
    model = ModelRunner(training=False, trial_path=model_name).model
    # shape_name = evaluation[model_name].keys()[0]
    # shape_evaluation = evaluation[model_name][shape_name]
    for shape_name, shape_evaluation in evaluation[model_name].items():
        angle = int(sn.get_metadata(shape_name)['augmentation'].numpy()[-3:])
        if not 250 < angle < 290:
            continue
        display_evaluation_for_shape(model, shape_name, shape_evaluation)

    # fits = plausiblility.load_plausibilities()[shape_name]


def display_evaluation_for_shape(model, shape_name, shape_evaluation):
    print("Loading valid fits")
    valid_fits = plausiblility.get_valid_fits(shape_name)
    # plausibles = [conversions.transform_voxelgrid(sn.get(name)['gt_occ'], T, scale=0.01) for name, T, _, _ in
    #               valid_fits]

    elem = sn.get(shape_name)
    e = {k: tf.expand_dims(v, axis=0) for k, v in elem.items()}
    VG_PUB.publish_elem(elem)
    num_particles = len(shape_evaluation['particle_distances'][0])

    print("Sampling particles for {}".format(shape_name))
    tf.random.set_seed(42)
    particles = [model(e)['predicted_occ'] for _ in range(num_particles)]

    best_particles = np.argmin(shape_evaluation['particle_distances'], axis=1)
    for i in range(len(list(best_particles))):
        VG_PUB.publish("plausible", plausibles[i])
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
