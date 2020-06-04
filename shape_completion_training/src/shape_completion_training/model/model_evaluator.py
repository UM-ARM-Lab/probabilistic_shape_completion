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


def get_plausibles(shape_name):
    sn = data_tools.get_addressible_shapenet()
    valid_fits = plausiblility.get_valid_fits(shape_name)
    plausibles = [conversions.transform_voxelgrid(sn.get(name)['gt_occ'], T, scale=0.01)
                  for name, T, _, _ in valid_fits]
    return plausibles


def sample_particles(model, input_elem, num_particles):
    tf.random.set_seed(42)
    return [model(input_elem)['predicted_occ'] for _ in range(num_particles)]


def compute_plausible_distances(ref_name, particles):
    plausibles = get_plausibles(ref_name)
    distances = [[chamfer_distance(a, b, scale=0.01, downsample=4).numpy()
                  for a in particles] for b in plausibles]
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
            particles = sample_particles(model, elem, num_particles)
            results = {}
            results["best_particle_iou"] = best_match_value(elem['gt_occ'], particles, metric=metrics.iou).numpy()
            results["best_particle_chamfer"] = \
                best_match_value(elem['gt_occ'], particles,
                                 metric=lambda a, b: chamfer_distance(a, b, scale=0.01, downsample=4)).numpy()
            results["particle_distances"] = compute_plausible_distances(elem_name, particles)
            all_metrics[elem_name] = results

    return all_metrics


