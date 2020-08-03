from __future__ import print_function
import rospy
import shape_completion_training.model.observation_model
from shape_completion_training.utils import data_tools
from shape_completion_visualization.voxelgrid_publisher import VoxelgridPublisher
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.model import plausiblility
import numpy as np
from PIL import Image

"""
This is to attempt to find a "good" observation model, that has a relatively close probability for 
2.5D views that I think should be close
"""



def show_image(observation, underlying_state):
    observed_depth = data_tools.simulate_depth_image(observation)
    expected_depth = data_tools.simulate_depth_image(underlying_state)
    # error = observed_depth - expected_depth
    # dx, dy = tf.image.image_gradients(tf.expand_dims(tf.expand_dims(expected_depth, 0), -1))
    # g = tf.maximum(tf.abs(dx), tf.abs(dy))
    # g = tf.cast(g > 10, tf.float32)
    # g = tf.nn.convolution(g, tf.ones([3,3,1,1]), padding='SAME')
    # g = tf.squeeze(g)
    # e = np.clip(np.abs(error.numpy()) * 255 * 1/10, 0, 255)
    # # g = tf.abs(tf.squeeze(dy))
    # e = g.numpy() * 255.
    # Image.fromarray(e).show()
    # mask = model_evaluator.mask_high_gradient(expected_depth)
    mask = shape_completion_training.model.observation_model.mask_empty(observed_depth, expected_depth)
    # Image.fromarray(mask.numpy()*255.).show()
    p = shape_completion_training.model.observation_model._observation_model(observation, underlying_state, 2)

    # g = p * (1-mask) + 1.0/20 * mask
    # g = g.numpy()
    g = p
    Image.fromarray(255 - np.clip(g * 255. , 0, 255)).show()


def run():
    print("Loading addressable shapenet")
    sn = data_tools.get_addressible_shapenet(use_train=False)
    name = "9737c77d3263062b8ca7a0a01bcd55b60.0_0.0_0.0_265"
    good_fit_name = "9737c77d3263062b8ca7a0a01bcd55b60.0_0.0_0.0_275"
    bad_fit_name = "9737c77d3263062b8ca7a0a01bcd55b60.0_0.0_0.0_300"

    reference = sn.get(name)
    good = sn.get(good_fit_name)
    bad = sn.get(bad_fit_name)

    print("Loading plausibilities")
    best_fits = plausiblility.load_plausibilities()[name]

    good_fitted = conversions.transform_voxelgrid(good['gt_occ'], best_fits[good_fit_name]['T'], scale=0.01)
    bad_fitted = conversions.transform_voxelgrid(bad['gt_occ'], best_fits[bad_fit_name]['T'], scale=0.01)

    VG_PUB.publish_elem(sn.get(name))
    show_image(good_fitted, reference['gt_occ'])
    VG_PUB.publish("predicted_occ", good_fitted)
    VG_PUB.publish("sampled_occ", bad_fitted)

    p_self = shape_completion_training.model.observation_model.observation_likelihood_geometric_mean(reference['gt_occ'], reference['gt_occ'])
    p_good = shape_completion_training.model.observation_model.observation_likelihood_geometric_mean(good_fitted, reference['gt_occ'],
                                                                                                     std_dev_in_voxels=1)
    p_bad = shape_completion_training.model.observation_model.observation_likelihood_geometric_mean(bad_fitted, reference['gt_occ'])

    print("self: {}, good: {}, bad: {}".format(p_self, p_good, p_bad))


def compare():
    sn = data_tools.get_addressible_dataset(dataset_name="shapenet", use_train=False)
    name = "9737c77d3263062b8ca7a0a01bcd55b60.0_0.0_0.0_265"
    best_fits = plausiblility.get_plausibilities_for(name, "shapenet")
    for i in range(30):
        info = best_fits[i]
        other_name, T, _, _ = best_fits[i]
        fitted = conversions.transform_voxelgrid(sn.get(other_name)['gt_occ'], T, scale=0.01)
        p = shape_completion_training.model.observation_model.observation_likelihood_geometric_mean(sn.get(name)['gt_occ'], fitted)
        oob = shape_completion_training.model.observation_model.out_of_range_count(sn.get(name)['gt_occ'], fitted)
        VG_PUB.publish("sampled_occ", fitted)
        print("{}: name = {}: oob = {}: p= {}".format(i, other_name, oob, p))


if __name__ == "__main__":
    VG_PUB = VoxelgridPublisher()
    rospy.init_node('wip_observation_model_publisher')
    rospy.loginfo("WIP Observation Model Publisher")
    run()
    compare()
    # rospy.spin()
