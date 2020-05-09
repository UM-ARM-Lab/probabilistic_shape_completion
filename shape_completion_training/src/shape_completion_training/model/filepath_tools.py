from __future__ import print_function

from datetime import datetime
import json
import pathlib
import subprocess
from os.path import join

import git


def unique_trial_name(*names):
    stamp = "{:%B_%d_%H-%M-%S}".format(datetime.now())
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    format_string = "{}_{}" + len(names) * '_{}'
    return format_string.format(stamp, sha, *names)


def create_or_load_trial(trial_name=None, params=None, trials_directory=None, write_summary=True):
    if '/' in trial_name:
        training = False
        # load, raises an exception if it doesn't exist
        full_trial_directory, params = load_trial(trial_name=trial_name, trials_directory=trials_directory)
    else:
        training = True
        full_trial_directory = create_trial(params=params,
                                            trial_name=trial_name,
                                            write_summary=write_summary,
                                            trials_directory=trials_directory)
    return full_trial_directory, params, training


def load_trial(trial_name=None, trials_directory=None):
    if trials_directory is None:
        r = rospkg.RosPack()
        shape_completion_training_path = pathlib.Path(r.get_path('shape_completion_training'))
        trials_directory = shape_completion_training_path / 'trials'
    trials_directory.mkdir(parents=True, exist_ok=True)

    # attempting to load existing trial
    full_directory = trials_directory / trial_name
    if not full_directory.is_dir():
        raise ValueError("Cannot load, this trial subdirectory does not exist")

    params_filename = full_directory / 'params.json'
    with params_filename.open("r") as params_file:
        params = json.load(params_file)
    return full_directory, params


def create_trial(params, trial_name=None, write_summary=True, trials_directory=None):
    """
    Returns the path to the directory for the trial, creates directories as needed
    """
    if trials_directory is None:
        r = rospkg.RosPack()
        shape_completion_training_path = pathlib.Path(r.get_path('shape_completion_training'))
        trials_directory = shape_completion_training_path / 'trials'
    trials_directory.mkdir(parents=True, exist_ok=True)

    if trial_name is None:
        # creating a new trial with tmp as the group name
        print("No trial_namename given. Using tmp")
        group_name = "tmp"
    else:
        group_name = trial_name

    # make subdirectory
    unique_trial_subdirectory_name = unique_trial_name()
    full_directory = trials_directory / group_name / unique_trial_subdirectory_name
    full_directory.mkdir(parents=True, exist_ok=False)
    # save params
    params_filename = full_directory / 'params.json'
    with params_filename.open("w") as params_file:
        json.dump(params, params_file, indent=2)
    # write summary
    if write_summary:
        _write_summary(full_directory, trial_name)
    return full_directory


def rm_tree(path):
    path = pathlib.Path(path)
    for child in path.glob('*'):
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    path.rmdir()


def _write_summary(full_trial_directory, trial_name):
    with (full_trial_directory / 'readme.txt').open("w") as f:
        f.write(datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
        f.write("\nTrial trial_nickname: {}\n".format(trial_name))

        f.write("git show --summary:\n")
        f.write(subprocess.check_output(['git', 'show', '--summary']))
        f.write("git status:\n")
        f.write(subprocess.check_output(['git', 'status']))
        f.write("git diff:\n")
        f.write(subprocess.check_output(['git', 'diff']))


def get_default_params():
    r = rospkg.RosPack()
    shape_completion_training_path = pathlib.Path(r.get_path('shape_completion_training'))
    default_params_filename = shape_completion_training_path / 'default_params.json'
    return json.load(default_params_filename.open('r'))
