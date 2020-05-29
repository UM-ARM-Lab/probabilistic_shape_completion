import rospkg
import pathlib
import pickle
from shape_completion_training.model import data_tools
from shape_completion_training.voxelgrid import fit, conversions
from shape_completion_training.model import model_evaluator
import progressbar


def _get_path():
    r = rospkg.RosPack()
    trial_path = pathlib.Path(r.get_path('shape_completion_training')) / "data" / \
                 "ShapeNetCore.v2_augmented" / "plausibility.pkl"
    return trial_path.as_posix()


def load_plausibilities():
    with open(_get_path(), "rb") as f:
        return pickle.load(f)


def save_plausibilities(plausibilities_dict):
    with open(_get_path(), "wb") as f:
        pickle.dump(plausibilities_dict, f)


def compute_icp_fit_dict(metadata):
    best_fits = {}
    num_shapes = 0
    for i in metadata:
        num_shapes += 1

    ds = data_tools.load_voxelgrids(metadata)
    ds = data_tools.simulate_input(ds, 0, 0, 0)

    widgets = [
        '  ', progressbar.Counter(), '/', str(num_shapes),
        ' ', progressbar.Variable("CurrentShape"), ' ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]

    with progressbar.ProgressBar(widgets=widgets, max_value=num_shapes) as bar:
        for i, reference in ds.enumerate():
            # bar.update(i.numpy())
            bar.update(i.numpy(), CurrentShape=data_tools.get_unique_name(reference))

            best_fit_for_reference = {}

            for other in ds:
                other_name = data_tools.get_unique_name(other)
                # print("    Fitting: {}".format(other_name))
                T = fit.icp_transform(other['known_occ'], reference['known_occ'], scale=0.01)
                fitted = conversions.transform_voxelgrid(other['gt_occ'], T, scale=0.01)
                p = model_evaluator.observation_likelihood_geometric_mean(reference['gt_occ'], fitted,
                                                                          std_dev_in_voxels=2)
                best_fit_for_reference[other_name] = {"T": T, "observation_probability": p}

            best_fits[data_tools.get_unique_name(reference)] = best_fit_for_reference
    return best_fits
