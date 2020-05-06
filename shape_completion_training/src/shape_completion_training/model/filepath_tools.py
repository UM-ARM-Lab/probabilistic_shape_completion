from __future__ import print_function
import os
import time
from os.path import join
import shutil
import datetime
import subprocess
import json

import git


def unique_trial_name(*names):
    now = str(int(time.time()))
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha[:10]
    format_string = "{}_{}" + len(names) * '_{}'
    return format_string.format(now, sha, *names)


def get_trial_directory(base_directory, nick=None, expect_reuse=False, write_summary=True):
    """
    Returns the filepath to the directory for the trial, prompting the user for information if necessary.
    The directory is created if it does not exist.

    Call as: get_trial_directory("/path/to/trials/directory/")
    """
    if not os.path.isdir(base_directory):
        _make_new_trials_directory(base_directory)

    if nick is None:
        nick = _make_tmp_nick(base_directory)

    unique_trial_subdirectory_name = unique_trial_name()
    fp = join(base_directory, nick, unique_trial_subdirectory_name)
    os.mkdir(fp)

    if write_summary:
        _write_summary(fp, nick)

    print("Running trial {} at {}".format(nick, fp))
    return fp


def handle_params(default_params_fp, model_params_fp, given_params):
    """
    Handles loading and saving of the params
    If params is None it will load from params from the directory
    Otherwise it save params to the directory
    Returns the params loaded or saved
    Prompts user if default parameters do not match the given parameters

    TODO: If given_params is specified this will silently overwrite any params already part of the model.
    """
    if given_params is None:
        return _load_params(default_params_fp, model_params_fp)

    ### Check defaults
    with open(join(default_params_fp, 'default_params.json'), 'r') as f:
        default_params = json.load(f)

    given_keys = set(given_params.keys())
    default_keys = set(default_params.keys())

    if given_keys != default_keys:
        print()
        print("!! Warning !!")
        print("Default params and given params have different keys. The defaults should have the same entry names as "
              "the specified params. This difference may prevent properly reloading after future changes!")
        for missing_default_key in given_keys - default_keys:
            print("{} missing from defaults".format(missing_default_key))
        print()
        for missing_given_key in default_keys - given_keys:
            print("{} missing from specified keys".format(missing_given_key))
        print("Press any key to continue")
        raw_input()

    _write_params(model_params_fp, given_params)
    return given_params


def _write_params(filepath, params_dict):
    with open(join(filepath, 'params.json'), 'w') as f:
        json.dump(params_dict, f, sort_keys=True)


def _load_params(default_params_fp, filepath):
    with open(join(default_params_fp, 'default_params.json'), 'r') as f:
        params = json.load(f)

    with open(join(filepath, 'params.json'), 'r') as f:
        params.update(json.load(f))
        return params


def _make_new_trials_directory(trials_directory):
    print("")
    print("WARNING: Trials directory does not exist, making it for you")
    os.mkdir(trials_directory)


def _make_tmp_nick(trials_directory):
    nick = "tmp"
    print("No nickname given. Using {}".format(nick))
    fp = join(trials_directory, nick)
    if not os.path.isdir(fp):
        os.mkdir(fp)
    return nick


def _write_summary(fp, nick, summary=None):
    with open(join(fp, "readme.txt"), "w") as f:
        f.write(datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S"))
        f.write("\nTrial nickname: {}\n".format(nick))

        if summary is None:
            print("Type a short summary for this trial")
            summary = raw_input()
        f.write("Summary: \n{}\n\n".format(summary))
        f.write("git show --summary:\n")
        f.write(subprocess.check_output(['git', 'show', '--summary']))
        f.write("git status:\n")
        f.write(subprocess.check_output(['git', 'status']))
        f.write("git diff:\n")
        f.write(subprocess.check_output(['git', 'diff']))
