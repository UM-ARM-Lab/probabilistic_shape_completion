#! /usr/bin/env python
import copy
from random import shuffle

from shape_completion_training.utils import data_tools
from shape_completion_training.model.modelrunner import ModelRunner
from bsaund_shape_completion import voxelgrid_publisher
import tensorflow as tf
import rospy
from shape_completion_training.model.utils import add_batch_to_dict, log_normal_pdf, stack_known
from shape_completion_training.voxelgrid.bounding_box import unflatten_bounding_box, flatten_bounding_box
import numpy as np
from matplotlib import pyplot as plt


def get_flow():
    # flow_trial = "Flow/July_02_10-47-22_d8d84f5d65"
    flow_trial = "FlowYCB/July_16_20-50-01_9d37e040d4"
    mr = ModelRunner(training=False, trial_path=flow_trial)

    return mr.model.flow


def view_flow():
    flow = get_flow()
    vg_pub = voxelgrid_publisher.VoxelgridPublisher()
    X = flow.distribution.sample(100)
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
    mr = ModelRunner(training=False, trial_path="NormalizingAE/July_02_15-15-06_ede2472d34")
    # mr = get_untrained_model()
    flow = get_flow()

    for i in range(0, 10000, 5):
        elem = sn.get(sn.train_names[i])
        elem = add_batch_to_dict(elem)
        vg_pub.publish_elem(elem)
        rospy.sleep(1)

        for i in range(10):
            output = mr.model(elem)
            # _, latent_box = mr.model.split_box(output['latent_mean'])
            _, latent_box = mr.model.split_box(output['sampled_latent'])
            flat_bb = flow.bijector.forward(latent_box)
            # flat_bb_1 = mr.model.flow.bijector.forward(latent_mean_box)
            # print(flat_bb - flat_bb_1)
            bb = unflatten_bounding_box(flat_bb)
            vg_pub.publish_bounding_box(bb)
            rospy.sleep(0.2)


        # l1 = mr.model.flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        # l2 = flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        # bb = unflatten_bounding_box(mr.model.flow.bijector.forward(l2.numpy()))
        # print("mean: {}".format(output['mean'].numpy()))
        # print("var: {}".format(tf.exp(output['logvar']).numpy()))




def view_augmented_ae():
    vg_pub = voxelgrid_publisher.VoxelgridPublisher()
    sn = data_tools.get_shapenet()
    mr = ModelRunner(training=False, trial_path="Augmented_VAE/May_21_20-00-00_0000000000")

    for i in range(0, 1000, 2):
        elem = sn.get(sn.train_names[i])
        elem = add_batch_to_dict(elem)
        mean, logvar = mr.model.encode(stack_known(elem))
        mean_f, mean_angle = mr.model.split_angle(mean)
        logvar_f, logvar_angle = mr.model.split_angle(logvar)
        vg_pub.publish_elem(elem)
        print("{} +/- {}".format(mean_angle.numpy()[0,0], tf.exp(logvar_angle).numpy()[0,0]))
        print("{} actually".format(elem['angle'].numpy()[0]))
        rospy.sleep(1)




def view_latent_space():
    sn = data_tools.get_shapenet()
    flow = get_flow()

    latents = []

    train_names = copy.deepcopy(sn.train_names)
    shuffle(train_names)
    for name in train_names[0:500]:
        elem = sn.get(name)
        elem = add_batch_to_dict(elem)
        l = flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        latents.append(l.numpy().tolist()[0])
    latents_array = np.array(latents)
    for i in range(0, 24, 2):
        plt.scatter(latents_array[:, i], latents_array[:, i + 1])
        plt.xlabel('latent[i]')
        plt.ylabel('latent[j]')
        plt.xlim([-20, 20])
        plt.ylim([-20, 20])
        plt.pause(0.05)
    plt.show()


def view_latent_space_as_movie():
    sn = data_tools.get_shapenet()
    flow = get_flow()

    train_names = copy.deepcopy(sn.train_names)
    # shuffle(train_names)

    color_ind = -1
    cur_id = "unlabeled"
    colors = ['b', 'r', 'g', 'm', 'c', 'k', 'y']

    for name in train_names[0:1000]:
        elem = sn.get(name)
        print(name)
        elem = add_batch_to_dict(elem)
        l = flow.bijector.inverse(flatten_bounding_box(elem['bounding_box']))
        # plt.scatter(l[0,0], l[0,1], c='b')
        #plt.scatter(l[0,0], elem['angle'][0])

        shape_id = elem['id'].numpy()[0]
        if shape_id != cur_id:
            cur_id = shape_id
            color_ind += 1

        latent_ind=4
        plt.scatter(elem['angle'][0], l[0, latent_ind], c=colors[color_ind])
        plt.xlim([-5, 365])
        plt.ylim([-5, 5])
        plt.xlabel("angle")
        plt.ylabel("latent[{}]".format(latent_ind))
        plt.pause(0.05)
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
        var = [.1] * 24
        logvar = tf.math.log(var)
        loss = -log_normal_pdf(l, mean, logvar)
        print(loss.numpy())


if __name__ == "__main__":
    rospy.init_node("bounding_box_flow_publisher")
    view_flow()
    # view_inferred_bounding_box()
    # view_latent_space()
    # view_latent_space_as_movie()
    # check_loss()
    # view_augmented_ae()

    # mr.train_and_test(sn.train_ds)
