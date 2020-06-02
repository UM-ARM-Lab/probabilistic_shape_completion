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

if __name__ == "__main__":
    rospy.init_node('evaluation_display')
    VG_PUB = VoxelgridPublisher()

    evaluation = model_evaluator.load_evaluation()
    print("Loading addressable shapenet")
    sn = data_tools.get_addressible_shapenet(use_train=False)

    model_name = evaluation.keys()[0]
    model = ModelRunner(training=False, trial_path=model_name).model

    shape_name = evaluation[model_name].keys()[0]
    # fits = plausiblility.load_plausibilities()[shape_name]
    print("Loading valid fits")
    valid_fits = plausiblility.get_valid_fits(shape_name)
    plausibles = [conversions.transform_voxelgrid(sn.get(name)['gt_occ'], T, scale=0.01) for name, T, _, _ in valid_fits]

    elem = sn.get(shape_name)
    e = {k: tf.expand_dims(v, axis=0) for k, v in elem.items()}
    VG_PUB.publish_elem(elem)
    info = evaluation[model_name][shape_name]
    num_particles = len(info['particle_distances'][0])

    tf.random.set_seed(42)
    particles = [model(e)['predicted_occ'] for _ in range(num_particles)]

    best_particles = np.argmin(info['particle_distances'], axis=1)
    for i in range(len(list(best_particles))):

        VG_PUB.publish("plausible", plausibles[i])
        VG_PUB.publish("predicted_occ", particles[best_particles[i]])
        rospy.sleep(1)


    print(evaluation)
