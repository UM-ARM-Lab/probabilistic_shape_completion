from __future__ import print_function

import json
import pathlib
import subprocess
from datetime import datetime

import git
import rospkg
from colorama import Fore


def get_shape_completion_package_path():
    """
    Get the path to shape_completion_training. Must be run in a Ros node
    @return: The pathlib.Path to shape_completion_training
    """
    r = rospkg.RosPack()
    return pathlib.Path(r.get_path('shape_completion_training'))


def make_unique_trial_subdirectory_name(*names):
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    format_string = "{}_{}" + len(names) * '_{}'
    return format_string.format(stamp, sha, *names)


def create_or_load_trial(group_name=None, trial_path=None, params=None, trials_directory=None, write_summary=True):
    """
    @param group_name:
    @type group_name: str
    @param trial_path:
    @type trial_path: str
    @param params:
    @type params: dict
    @param trials_directory:
    @type trials_directory: str
    @param write_summary: Only applies when creating a new trial
    @type write_summary: bool
    @return: trial_path, trial_hyper_parameters
    @rtype: (pathlib.Path, dict)
    """
    if trial_path is not None:
        # Resume and warn for errors
        if group_name is not None:
            msg = "Ignoring group_name {} and resuming from trial_path {}".format(group_name, trial_path)
            print(Fore.YELLOW + msg + Fore.RESET)
        if params is not None:
            print(Fore.YELLOW + "Ignoring params, loading existing ones" + Fore.RESET)
        return load_trial(trial_path)
    elif group_name is not None:
        return create_trial(group_name, params, trials_directory, write_summary)
    else:
        print("No group name specified: using 'tmp' trial")
        return create_trial('tmp', params, trials_directory, write_summary)


def load_trial(trial_path):
    """
    @param trial_path: full path or relative path from shape_completion_training/trials
    @type trial_path: str
    @return:
    """
    trial_path = pathlib.Path(trial_path)
    if not trial_path.is_absolute():
        r = rospkg.RosPack()
        trial_path = pathlib.Path(r.get_path('shape_completion_training')) / "trials" / trial_path
    if not trial_path.is_dir():
        raise ValueError("Cannot load, the path {} is not an existing directory".format(trial_path))

    params_filename = trial_path / 'params.json'
    with params_filename.open("r") as params_file:
        params = json.load(params_file)
    return trial_path, params


def create_trial(group_name, params, trials_directory=None, write_summary=True):
    if trials_directory is None:
        r = rospkg.RosPack()
        shape_completion_training_path = pathlib.Path(r.get_path('shape_completion_training'))
        trials_directory = shape_completion_training_path / 'trials'
    if not trials_directory.exists():
        trials_directory.mkdir(parents=True)

    # make subdirectory
    unique_trial_subdirectory_name = make_unique_trial_subdirectory_name()
    full_directory = trials_directory / group_name / unique_trial_subdirectory_name
    if not full_directory.exists():
        full_directory.mkdir(parents=True)

    # save params
    params_filename = full_directory / 'params.json'
    with open(params_filename.as_posix(), "w") as params_file:
        json.dump(params, params_file, indent=2)

    # write summary
    if write_summary and group_name is not None:
        _write_summary(full_directory, group_name, unique_trial_subdirectory_name)
    return full_directory, params


def rm_tree(path):
    path = pathlib.Path(path)
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


def _write_summary(full_trial_directory, group_name, unique_trial_subdirectory_name):
    with open((full_trial_directory / 'readme.txt').as_posix(), "w") as f:
        f.write(datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
        f.write("\nTrial trial_nickname: {}/{}\n".format(group_name, unique_trial_subdirectory_name))

        f.write("git show --summary:\n")
        f.write(subprocess.check_output(['git', 'show', '--summary']))
        f.write("git status:\n")
        f.write(subprocess.check_output(['git', 'status']))
        f.write("git diff:\n")
        f.write(subprocess.check_output(['git', 'diff']))


