from shape_completion_training.voxelgrid import metrics
from shape_completion_training.voxelgrid.metrics import best_match_value, chamfer_distance
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.model import data_tools
from shape_completion_training.model import plausiblility
import rospkg
import pickle
import progressbar
import pathlib
import tensorflow as tf


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


def compute_plausible_distances(ref_name, particles):
    sn = data_tools.get_addressible_shapenet()
    fits = plausiblility.load_plausibilities()[ref_name]
    valid_fits = plausiblility.get_valid_fits(ref_name)
    plausibles = [conversions.transform_voxelgrid(sn.get(name)['gt_occ'], T) for name, T, _, _ in valid_fits]

    distances = [[chamfer_distance(a, b, scale=0.01, downsample=4).numpy() for a in particles] for b in plausibles]
    return distances


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
            elem_name = data_tools.get_unique_name(elem)[0]
            bar.update(i.numpy(), CurrentShape=elem_name)
            results = {}
            tf.random.set_seed(42)
            particles = [model(elem)['predicted_occ'] for _ in range(num_particles)]
            results["best_particle_iou"] = best_match_value(elem['gt_occ'], particles, metric=metrics.iou)
            results["best_particle_chamfer"] = best_match_value(elem['gt_occ'], particles,
                                                                metric=lambda a, b: chamfer_distance(a, b, scale=0.01,
                                                                                                     downsample=4))
            results["particle_distances"] = compute_plausible_distances(elem_name, particles)
            all_metrics[elem_name] = results

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
