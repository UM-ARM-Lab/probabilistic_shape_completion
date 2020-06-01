import tensorflow as tf
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.model import data_tools
from shape_completion_training.model import utils
import tensorflow_probability as tfp
import rospkg
import pickle
import progressbar
import pathlib


def observation_likelihood(observation, underlying_state, std_dev_in_voxels=1):
    return tf.reduce_prod(_observation_model(observation, underlying_state, std_dev_in_voxels))


def observation_likelihood_geometric_mean(observation, underlying_state, std_dev_in_voxels=1):
    return utils.reduce_geometric_mean(_observation_model(observation, underlying_state, std_dev_in_voxels))


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


def out_of_range_count(observation, underlying_state, width=4):
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

    return tf.reduce_sum(1-p)


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


def _get_path():
    r = rospkg.RosPack()
    trial_path = pathlib.Path(r.get_path('shape_completion_training')) / "trials" / "evaluation.pkl"
    return trial_path.as_posix()


def load_evaluation():
    with open(_get_path(), "rb") as f:
        return pickle.load(f)


def save_evaluation(evaluation_dict):
    with open(_get_path(), "wb") as f:
        pickle.dump(evaluation_dict, f)


def evaluate_model(model, test_set, test_set_size, num_particles=100):
    all_metrics = {}

    widgets = [
        '  ', progressbar.Counter(), '/', str(test_set_size),
        ' ', progressbar.Variable("CurrentShape"), ' ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]
    with progressbar.ProgressBar(widgets=widgets, max_value=test_set_size) as bar:
        for i, elem in test_set.batch(1).enumerate():
            # print("Evaluating {}".format(data_tools.get_unique_name(elem)))
            bar.update(i.numpy(), CurrentShape=data_tools.get_unique_name(elem)[0])
            results = {}
            particles = [model(elem)['predicted_occ'] for _ in range(num_particles)]
            results["best_particle_iou"] = metrics.best_match_value(elem['gt_occ'], particles, metric=metrics.iou)
            all_metrics[data_tools.get_unique_name(elem)[0]] = results
    return all_metrics


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_element(self, model_input, num_samples=100):
        return DatumEvaluator(self.model, model_input, num_samples)


class DatumEvaluator:
    def __init__(self, model, model_input, num_samples):
        self.model = model
        self.model_input = model_input
        self.particles = self.sample_particles(num_samples)

    def sample_particles(self, num_samples):
        return [self.model(self.model_input)['predicted_occ'] for _ in range(num_samples)]

    def get_best_particle(self, metric=metrics.iou):
        def score(vg):
            return metric(self.model_input['gt_occ'], vg)

        return max(self.particles, key=score)

    def get_plausible_samples(self, dataset):
        pass

    def fit_best_sample(self, particle_ind, dataset, num_trials):
        vg = self.particles[particle_ind]

        def m(vg1, vg2):
            vg_fit = fit.icp(vg2, vg1, scale=0.1, max_iter=10, downsample=2)
            return -metrics.chamfer_distance(vg1, vg_fit, scale=0.1, downsample=2)
