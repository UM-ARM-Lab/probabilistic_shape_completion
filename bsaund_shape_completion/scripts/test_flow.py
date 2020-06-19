#! /usr/bin/env python
import copy
from random import shuffle

from shape_completion_training.model import data_tools
from shape_completion_training.model.modelrunner import ModelRunner
from bsaund_shape_completion import voxelgrid_publisher
import tensorflow as tf
import rospy
from shape_completion_training.model.utils import add_batch_to_dict, log_normal_pdf
from shape_completion_training.voxelgrid.bounding_box import unflatten_bounding_box, flatten_bounding_box
import numpy as np
from matplotlib import pyplot as plt


def get_flow():
    mr = ModelRunner(training=False, trial_path="Flow/June_13_13-09-11_4bef25fbe3")
    return mr.model.flow


def view_flow():
    flow = get_flow()
    vg_pub = voxelgrid_publisher.VoxelgridPublisher()
    X = flow.distribution.sample(10)
    Y = flow.bijector.forward(X)
    for bb_flat in Y:
        bb = tf.reshape(bb_flat, (8, 3))
        print(bb.numpy())
        vg_pub.publish_bounding_box(bb)
        rospy.sleep(1)

    print("done")


def get_untrained_model():
    params = {
        'num_latent_layers': 24,
        'translation_pixel_range_x': 0,
        'translation_pixel_range_y': 0,
        'translation_pixel_range_z': 0,
        'simulate_partial_completion': False,
        'simulate_random_partial_completion': False,
        # 'network': 'VoxelCNN',
        # 'network': 'VAE_GAN',
        # 'network': 'Augmented_VAE',
        # 'network': 'Conditional_VCNN',
        'network': 'NormalizingAE',
        'batch_size': 16,
        'learning_rate': 1e-3,

    }
    mr = ModelRunner(training=False, params=params)
    return mr


def view_inferred_bounding_box():
    vg_pub = voxelgrid_publisher.VoxelgridPublisher()
    sn = data_tools.get_shapenet()
    mr = ModelRunner(training=False, trial_path="Normalizing_AE/June_18_20-47-38_8394515bf0")
    # mr = get_untrained_model()
    flow = get_flow()

    for i in range(100):
        elem = sn.get(sn.train_names[i])
        elem = add_batch_to_dict(elem)
        output = mr.model(elem)
        flat_bb = flow.bijector.forward(output['mean'])
        flat_bb_1 = mr.model.flow.bijector.forward(output['mean'])
        print(flat_bb - flat_bb_1)
        bb = unflatten_bounding_box(flat_bb)
        vg_pub.publish_elem(elem)
        rospy.sleep(1)

        l1 = mr.model.flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        l2 = flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        # bb = unflatten_bounding_box(mr.model.flow.bijector.forward(l2.numpy()))
        print("mean: {}".format(output['mean'].numpy()))
        print("var: {}".format(tf.exp(output['logvar']).numpy()))

        vg_pub.publish_bounding_box(bb)
        rospy.sleep(1)


def view_latent_space():
    sn = data_tools.get_shapenet()
    flow = get_flow()

    latents = []

    train_names = copy.deepcopy(sn.train_names)
    shuffle(train_names)
    for name in train_names[0:1000]:
        elem = sn.get(name)
        elem = add_batch_to_dict(elem)
        l = flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        latents.append(l.numpy().tolist()[0])
    latents_array = np.array(latents)
    for i in range(0, 24, 2):
        plt.scatter(latents_array[:,i], latents_array[:,i+1])
        plt.show()


def check_loss():
    sn = data_tools.get_shapenet()
    flow = get_flow()

    latents = []

    train_names = copy.deepcopy(sn.train_names)
    shuffle(train_names)
    for name in train_names[0:1]:
        elem = sn.get(name)
        elem = add_batch_to_dict(elem)
        l = flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        # mean=[0.]*24
        mean = l
        var = [.1]*24
        logvar = tf.math.log(var)
        loss = -log_normal_pdf(l, mean, logvar)
        print(loss.numpy())


if __name__ == "__main__":
    rospy.init_node("bounding_box_flow_publisher")
    # view_flow()
    # view_inferred_bounding_box()
    view_latent_space()
    # check_loss()

    # mr.train_and_test(sn.train_ds)
