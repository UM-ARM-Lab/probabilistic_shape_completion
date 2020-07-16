import multiprocessing as mp
from shape_completion_training.utils import data_tools
import tensorflow as tf

NUM_PROCESSES = 3


def process_in_threads(target, args, num_threads):
    threads = []
    for _ in range(num_threads):
        thread = mp.Process(target=target, args=args)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def compute_single_best_fit(queue):
    """
    @param queue:
    @type queue: mp.Queue
    @return:
    """
    print("Running process")
    while True:
        reference, other = queue.get(block=True, timeout=100.0)
        if reference is None:
            return
        print("{}, {}".format(data_tools.get_unique_name(reference), data_tools.get_unique_name(other)))


def compute_best_fits(reference, ds):
    print("Process started")
    for other in ds:
        # try:
        #     offset = queue.get(False)
        # except Queue.Empty:
        #     time.sleep(1)
        #     print("Queue empty, returning")
        #     return

        # print("Skipping by {}".format(offset))
        # single_elem_ds = ds.skip(offset).take(1)
        # print("Getting single element")
        # other = next(single_elem_ds.__iter__())
        other = data_tools.load_voxelgrids_for_elem(other)

        print("Processing {} and {}".format(data_tools.get_unique_name(reference),
                                            data_tools.get_unique_name(other)))


def compute_plausibilities_in_parallel(metadata):
    best_fits = {}
    num_shapes = 0
    print("counting metadata")
    for i in metadata:
        num_shapes += 1
    print("counted")

    # ds = data_tools.load_voxelgrids(metadata)
    # ds = data_tools.simulate_input(ds, 0, 0, 0)
    options = tf.data.Options()
    options.experimental_optimization.autotune = False
    ds = metadata.with_options(options)

    for i, reference in ds.enumerate():
        q = mp.Queue()
        for j in range(num_shapes):
            q.put(j)

        threads = []
        for proc_num in range(NUM_PROCESSES):
            non_tf_ds = []
            for e in ds.shard(NUM_PROCESSES, proc_num):
                non_tf_ds.append(e)

            process = mp.Process(target=compute_best_fits, args=(reference, non_tf_ds))
            process.start()
            threads.append(process)
        for thread in threads:
            thread.join()
        print("")
        # time.sleep(1)

        # print("{}: {}".format(i, data_tools.get_unique_name(reference)))


if __name__ == "__main__":
    # train_dataset, test_dataset = data_tools.load_shapenet_metadata(shuffle=False)

    test_dataset = data_tools._load_metadata_train_or_test(shapes="all", shuffle=False, prefix="test")
    test_dataset = test_dataset.take(10)

    compute_plausibilities_in_parallel(test_dataset)

    # fits = plausiblility.compute_icp_fit_dict(test_dataset)
    # plausiblility.save_plausibilities(fits)
    #
    # loaded_fits = plausiblility.load_plausibilities()
    # print("Finished computing plausibilities")
