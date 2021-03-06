from __future__ import print_function
import Queue
import multiprocessing as mp
import os
import subprocess
import sys
from itertools import izip_longest

from shape_completion_training.utils import dataset_storage, obj_tools

NUM_THREADS_PER_CATEGORY = 5
NUM_THREADS_PER_OBJECT = 6
HARDCODED_BOUNDARY = '-bb -1.0 -1.0 -1.0 1.0 1.0 1.0'
NUM_THREADS = NUM_THREADS_PER_CATEGORY * NUM_THREADS_PER_OBJECT


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def process_in_threads(target, args, num_threads):
    threads = []
    for _ in range(num_threads):
        thread = mp.Process(target=target, args=args)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def binvox_object_file(fp):
    """
    Runs binvox on the input obj file
    """
    # TODO Hardcoded binvox path
    binvox_str = "~/useful_scripts/binvox -dc -pb -down -down -dmin 2 {} {}".format(HARDCODED_BOUNDARY, fp.as_posix())

    # Fast but inaccurate
    wire_binvox_str = "~/useful_scripts/binvox -e -pb -down -down -dmin 1 {} {}".format(HARDCODED_BOUNDARY,
                                                                                        fp.as_posix())
    # cuda_binvox_str = "~/useful_scripts/cuda_voxelizer -s 64 -f {}".format(fp)

    with open(os.devnull, 'w') as FNULL:
        subprocess.call(binvox_str, shell=True, stdout=FNULL)
        fp.with_suffix('.binvox').rename(fp.with_suffix(".mesh.binvox"))

        subprocess.call(wire_binvox_str, shell=True, stdout=FNULL)
        fp.with_suffix('.binvox').rename(fp.with_suffix(".wire.binvox"))

        # subprocess.call(cuda_binvox_str, shell=True, stdout=FNULL)

    file_dir, file_name = fp.parent.as_posix(), fp.stem
    augmentation = file_name[len('model_augmented_'):]
    gt = dataset_storage.load_gt_voxels_from_binvox(file_dir, augmentation)
    dataset_storage.save_gt_voxels(fp.with_suffix(".pkl"), gt, compression="gzip")


def binvox_object_file_worker(queue):
    while True:
        try:
            fp = queue.get(False)
        except Queue.Empty:
            return
        binvox_object_file(fp)


def augment_category(object_path, models_dirname="models", obj_filename="model_normalized.obj"):
    # shape_ids = ['a1d293f5cc20d01ad7f470ee20dce9e0']
    # shapes = ['214dbcace712e49de195a69ef7c885a4']
    shape_ids = [f.name for f in object_path.iterdir() if f.stem != "tfrecords"]
    shape_ids.sort()

    q = mp.Queue()
    for elem in zip(range(1, len(shape_ids) + 1), shape_ids):
        q.put(elem)

    print("")
    print("Augmenting shapes using {} threads".format(NUM_THREADS))
    print("Progress may appear eratic due to threading")
    print("")

    process_in_threads(target=augment_shape_worker, args=(q, object_path, models_dirname,
                                                          obj_filename, len(shape_ids),),
                       num_threads=NUM_THREADS_PER_CATEGORY)


def augment_shape_worker(queue, object_path, models_dirname, obj_filename, total):
    while True:
        try:
            count, shape_id = queue.get(False)
        except Queue.Empty:
            return

        sys.stdout.write('\033[2K\033[1G')
        print("{:03d}/{} Augmenting {}".format(count, total, shape_id), end="")
        sys.stdout.flush()
        fp = object_path / shape_id / models_dirname
        augment_shape(fp, obj_filename)


def augment_shape(filepath, obj_filename):
    """
    Augments the model at the filepath

    Augmentation involves rotating the model and converting all rotations to .binvox files

    @param filepath: pathlib.Path filepath, ending with the "models" folder
    @return: None
    """

    fp = filepath

    if fp is None:
        return

    old_files = [f for f in fp.iterdir() if f.name.startswith("model_augmented")]
    for f in old_files:
        f.unlink()

    obj_path = fp / obj_filename
    # print("Augmenting {}".format(fp))
    obj_tools.augment(obj_path.as_posix())

    augmented_obj_files = [f for f in fp.iterdir()
                           if f.name.startswith('model_augmented')
                           if f.name.endswith('.obj')]
    augmented_obj_files.sort()

    q = mp.Queue()
    for f in augmented_obj_files:
        # binvox_object_file(join(fp, f))
        q.put(f)
    process_in_threads(target=binvox_object_file_worker, args=(q,),
                       num_threads=NUM_THREADS_PER_OBJECT)

    # Cleanup large model files
    old_files = [f for f in fp.iterdir()
                 if f.name.startswith("model_augmented")
                 if not f.name.endswith(".pkl.gzip")]
    for f in old_files:
        f.unlink()