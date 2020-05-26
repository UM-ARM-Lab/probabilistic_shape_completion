import tensorflow as tf
from shape_completion_training.voxelgrid import metrics
from shape_completion_training.model import data_tools
import tensorflow_probability as tfp


def observation_probability(observation, underlying_state, std_dev_in_voxels = 1):
    observed_depth = data_tools.simulate_depth_image(observation)
    expected_depth = data_tools.simulate_depth_image(underlying_state)
    error = observed_depth - expected_depth
    depth_probs = tfp.distributions.Normal(0, std_dev_in_voxels).prob(error)
    return tf.reduce_prod(depth_probs)

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
