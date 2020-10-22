import tensorflow as tf
import tensorflow_probability as tfp

from shape_completion_training.utils import data_tools, tf_utils


def observation_likelihood(observation, underlying_state, std_dev_in_voxels=1):
    return tf.reduce_prod(_observation_model(observation, underlying_state, std_dev_in_voxels))


def observation_likelihood_geometric_mean(observation, underlying_state, std_dev_in_voxels=1):
    return tf_utils.reduce_geometric_mean(_observation_model(observation, underlying_state, std_dev_in_voxels))


def mask_high_gradient(expected_depth, gradient_threshold=10, inflation=3):
    dx, dy = tf.image.image_gradients(tf.expand_dims(tf.expand_dims(expected_depth, 0), -1))
    mask = tf.maximum(tf.abs(dx), tf.abs(dy))
    mask = tf.cast(mask > 10, tf.float32)
    mask = tf.nn.convolution(mask, tf.ones([3, 3, 1, 1]), padding='SAME')
    mask = tf.clip_by_value(tf.squeeze(mask), 0.0, 1.0)
    return mask


def mask_empty(expected_depth, observed_depth, max_depth=64):
    return tf.cast(tf.logical_and(observed_depth == max_depth, expected_depth == max_depth), tf.float32)


def range_likelihood(error, width):
    return tf.cast(tf.abs(error) < width, tf.float32)


def out_of_range_count(observation, underlying_state, width=4, additional_mask=None):
    """
    return the number of voxels in the observation that are out of the specified range
    given an underlying state
    @param observation:
    @param underlying_state:
    @param range:
    @return:
    """
    observed_depth = data_tools.simulate_depth_image(observation)
    expected_depth = data_tools.simulate_depth_image(underlying_state)
    error = observed_depth - expected_depth
    range_probs = range_likelihood(error, width)
    p = range_probs

    mask = mask_high_gradient(expected_depth)
    p = p * (1 - mask) + mask
    if additional_mask is not None:
        p = p * (1 - additional_mask) + additional_mask

    return tf.reduce_sum(1 - p)


def _observation_model(observation, underlying_state, std_dev_in_voxels, max_depth=64):
    observed_depth = data_tools.simulate_depth_image(observation)
    expected_depth = data_tools.simulate_depth_image(underlying_state)
    error = observed_depth - expected_depth
    # error = conversions.format_voxelgrid(error, True, True)
    # error = -1 * tf.nn.max_pool(-1 * error, ksize=5, strides=1, padding="VALID")
    # error = conversions.format_voxelgrid(error, False, False)

    depth_probs = tfp.distributions.Normal(0, std_dev_in_voxels).prob(error)

    """Mask out high gradient areas, set to constant probability"""
    mask = mask_high_gradient(expected_depth)
    depth_probs = depth_probs * (1 - mask) + (1.0 / max_depth) * mask

    """Mask out correclty predicted empty space. Set to 1.0"""
    mask = mask_empty(observed_depth, expected_depth)
    depth_probs = depth_probs * (1 - mask) + mask

    """Add small uniform support everywhere"""
    alpha = 0.01
    depth_probs = depth_probs * (1 - alpha) + (1.0 / max_depth) * alpha

    return depth_probs