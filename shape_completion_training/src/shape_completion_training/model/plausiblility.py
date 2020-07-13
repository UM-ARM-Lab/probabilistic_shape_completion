import rospkg
import pathlib
import pickle

from shape_completion_training.model.observation_model import observation_likelihood_geometric_mean, out_of_range_count
from shape_completion_training.model import data_tools
from shape_completion_training.voxelgrid import fit, conversions
from shape_completion_training.model.utils import memoize
from shape_completion_training.model.shapenet_storage import shapenet_load_path
import progressbar


def _get_path(identifier=""):
    r = rospkg.RosPack()
    trial_path = shapenet_load_path / "plausibility" + identifier + ".pkl"
    return trial_path.as_posix()


@memoize
def load_plausibilities():
    with open(_get_path(), "rb") as f:
        return pickle.load(f)


def save_plausibilities(plausibilities_dict, identifier=""):
    with open(_get_path(identifier), "wb") as f:
        pickle.dump(plausibilities_dict, f)


def get_valid_fits(name):
    fits = load_plausibilities()[name]
    valid_fits = [(k, v["T"], v["observation_probability"], v["out_of_range_count"])
                  for k, v in fits.items()
                  if v["out_of_range_count"] == 0]
    return valid_fits


def get_fits_for(name):
    """
    @param name:
    @return: sorted list of (other_name, T, observation_probability, out_of_range_count)
    """
    fits = load_plausibilities()[name]
    return sorted([(k, v["T"], v["observation_probability"], v["out_of_range_count"]) for k, v in fits.items()],
                  key=lambda x: x[2],
                  reverse=True)


def compute_icp_fit_dict(metadata):
    return compute_partial_icp_fit_dict(metadata, metadata)


def compute_partial_icp_fit_dict(reference_metadata, other_metadata):
    best_fits = {}
    num_shapes = 0
    for i in reference_metadata:
        num_shapes += 1

    reference_ds = data_tools.load_voxelgrids(reference_metadata)
    reference_ds = data_tools.simulate_input(reference_ds, 0, 0, 0)

    other_ds = data_tools.load_voxelgrids(other_metadata)
    other_ds = data_tools.simulate_input(other_ds, 0, 0, 0)

    widgets = [
        '  ', progressbar.Counter(), '/', str(num_shapes),
        ' ', progressbar.Variable("CurrentShape"), ' ',
        progressbar.Bar(),
        ' (', progressbar.ETA(), ') ',
    ]

    with progressbar.ProgressBar(widgets=widgets, max_value=num_shapes) as bar:
        for i, reference in reference_ds.enumerate():
            # bar.update(i.numpy())
            bar.update(i.numpy(), CurrentShape=data_tools.get_unique_name(reference))

            best_fit_for_reference = {}

            for other in other_ds:
                other_name = data_tools.get_unique_name(other)
                # print("    Fitting: {}".format(other_name))
                T = fit.icp_transform(other['known_occ'], reference['known_occ'], scale=0.01)
                fitted = conversions.transform_voxelgrid(other['gt_occ'], T, scale=0.01)
                p = observation_likelihood_geometric_mean(reference['gt_occ'], fitted,
                                                          std_dev_in_voxels=2)
                oob = out_of_range_count(reference['gt_occ'], fitted, width=4)
                best_fit_for_reference[other_name] = {"T": T, "observation_probability": p,
                                                      "out_of_range_count": oob}

            best_fits[data_tools.get_unique_name(reference)] = best_fit_for_reference
    return best_fits
