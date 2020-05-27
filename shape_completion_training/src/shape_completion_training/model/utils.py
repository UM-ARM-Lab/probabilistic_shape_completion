"""Some useful Utils for tensorflow"""

import os
import re
import subprocess

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23
import tensorflow as tf


def run_command(cmd):
    """Run command, return output as string."""
    output = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).communicate()[0]
    return output.decode("ascii")


def list_available_gpus():
    """Returns list of available GPU ids."""
    output = run_command("nvidia-smi -L")
    # lines of the form GPU 0: TITAN X
    if output == "":
        return []

    gpu_regex = re.compile(r"GPU (?P<gpu_id>\d+):")
    result = []
    for line in output.strip().split("\n"):
        m = gpu_regex.match(line)
        assert m, "Couldnt parse " + line
        result.append(int(m.group("gpu_id")))
    return result


def gpu_memory_map():
    """Returns map of GPU id to memory allocated on that GPU."""

    output = run_command("nvidia-smi")
    if output == "":
        return {}

    gpu_output = output[output.find("GPU Memory"):]
    # lines of the form
    # |    0      8734    C   python                                       11705MiB |
    memory_regex = re.compile(r"[|]\s+?(?P<gpu_id>\d+)\D+?(?P<pid>\d+).+[ ](?P<gpu_memory>\d+)MiB")
    rows = gpu_output.split("\n")
    result = {gpu_id: 0 for gpu_id in list_available_gpus()}
    for row in gpu_output.split("\n"):
        m = memory_regex.search(row)
        if not m:
            continue
        gpu_id = int(m.group("gpu_id"))
        gpu_memory = int(m.group("gpu_memory"))
        result[gpu_id] += gpu_memory
    return result


def pick_gpu_lowest_memory():
    """Returns GPU with the least allocated memory"""

    memory_gpu_map = [(memory, gpu_id) for (gpu_id, memory) in gpu_memory_map().items()]
    if len(memory_gpu_map) == 0:
        return ""
    best_memory, best_gpu = sorted(memory_gpu_map)[0]
    return best_gpu


def set_gpu_with_lowest_memory():
    best_gpu = str(pick_gpu_lowest_memory())
    if best_gpu == "":
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = best_gpu


def reduce_mean_dict(dict):
    reduced_dict = {}
    for k, v in dict.items():
        reduced_dict[k] = tf.reduce_mean(v)
    return reduced_dict


def sequence_of_dicts_to_dict_of_sequences(seq_of_dicts):
    # TODO: make a data structure that works both ways, as a dict and as a list
    dict_of_seqs = {}
    for d in seq_of_dicts:
        for k, v in d.items():
            if k not in dict_of_seqs:
                dict_of_seqs[k] = []
            dict_of_seqs[k].append(v)

    return dict_of_seqs


def reduce_geometric_mean(tensor):
    return tf.exp(tf.reduce_mean(tf.math.log(tensor)))

