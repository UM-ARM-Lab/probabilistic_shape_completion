
from __future__ import print_function
import os
from os.path import join
import shutil
import datetime
import subprocess





"""
Returns the filepath to the directory for the trial, prompting the user for information if necessary.
The directory is created if it does not exist.

Call as: get_trial_directory("/path/to/trials/directory/")
"""
def get_trial_directory(base_directory, nick=None):
    if not os.path.isdir(base_directory):
        _make_new_trials_directory(base_directory)

    if nick is None:
        print("Type a nickname for this trial")
        nick = raw_input()

    if nick == "":
        nick = _make_tmp_nick(base_directory)

    fp = join(base_directory, nick)
    reusing = _check_reuse_existing_directory(fp, nick)

    if not reusing:
        _write_summary(fp, nick)

    print("Running trial {} at {}".format(nick, fp))
    return fp



def _make_new_trials_directory(trials_directory):
    print("")
    print("WARNING: Trials directory does not exist")
    print("You have set the trials directory to {}, yet this directory does not exist".format(trials_directory))
    print("Create this new directory? y/N")
    fb = raw_input()

    if fb.lower() == 'y':
        os.mkdir(trials_directory)
        return

    raise RuntimeError("Trials directory does not exist: {}".format(trials_directory))


def _make_tmp_nick(trials_directory):
    nick = "tmp"
    print("No nickname given. Using {}".format(nick))
    fp = join(trials_directory, nick)
    if os.path.isdir(fp):
        print("Trial {} already exists. Delete and start fresh? y/N".format(nick))
        fb = raw_input()
        if fb.lower() == 'y':
            shutil.rmtree(fp)
    return nick


def _check_reuse_existing_directory(fp, nick):
    if not os.path.isdir(fp):
        os.mkdir(fp)
        return False
        
    print("Trial {} already exists. Load from existing? Y/n".format(nick))
    if raw_input().lower() == 'n':
        print("If you want to use this nick you must manually move (or delete) the existing trial")
        print()
        raise RuntimeError("No load directory specified. Trial {} exists and not reusing".format(nick))
    return True

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
    



