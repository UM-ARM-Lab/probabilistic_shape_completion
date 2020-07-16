"""Some useful Utils for tensorflow"""

import os
import re
import subprocess
import functools

# Nvidia-smi GPU memory parsing.
# Tested on nvidia-smi 370.23
import numpy as np
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


def add_batch_to_dict(elem):
    return {k: tf.expand_dims(v, axis=0) for k, v in elem.items()}


def remove_batch_from_dict(elem):
    """removes the first dimension of elem, assuming it is of size 1"""
    return {k: tf.squeeze(v, axis=0) for k, v in elem.items()}


def numpyify(elem):
    return {k: v.numpy() for k, v in elem.items()}


def reduce_geometric_mean(tensor):
    return tf.exp(tf.reduce_mean(tf.math.log(tensor)))


def memoize(func):
    cache = func.cache = {}

    @functools.wraps(func)
    def memoized_func(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]

    return memoized_func


def stack_known(inp):
    return tf.concat([inp['known_occ'], inp['known_free']], axis=4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    """
    Computes the probability density for a sample (vector) given a gaussian means and variance.
    @param sample: vector (batch x length)
    @param mean: vector (batch x vector_length)
    @param logvar: vector - log of variance of gaussian
    @param raxis: axis which to sum
    @return:
    """
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)