#! /usr/bin/env python
from __future__ import print_function

import tensorflow as tf

from shape_completion_training.utils import shapenet_storage
from shape_completion_training.utils import ycb_storage
from shape_completion_training.utils.dataset_storage import load_gt_only
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.model.utils import memoize
import numpy as np
import pickle


def simulate_omniscient_input(gt):
    """
    Given a single ground truth mask occupied list,
    return ground truth occupied and free,
    as well as simulate the known occupied and free
    """
    return gt, 1.0 - gt


def simulate_2_5D_input(gt):
    gt_occ = gt
    gt_free = 1.0 - gt
    known_occ = gt + 0.0
    known_free = gt_free + 0.0
    unknown_mask = np.zeros((gt.shape[1], gt.shape[2]))
    for h in range(gt.shape[1]):
        known_occ[h, :, :, 0] = np.clip(known_occ[h, :, :, 0] - unknown_mask, 0, 1)
        known_free[h, :, :, 0] = np.clip(known_free[h, :, :, 0] - unknown_mask, 0, 1)
        unknown_mask = unknown_mask + gt_occ[h, :, :, 0]
        unknown_mask = np.clip(unknown_mask, 0, 1)
    return known_occ, known_free


def simulate_slit_occlusion(known_occ, known_free, slit_zmin, slit_zmax):
    known_occ[:, :, 0:slit_zmin, 0] = 0
    known_free[:, :, 0:slit_zmin, 0] = 0

    known_occ[:, :, slit_zmax:, 0] = 0
    known_free[:, :, slit_zmax:, 0] = 0
    return known_occ, known_free


def get_slit_occlusion_2D_mask(slit_min, slit_width, mask_shape):
    slit_max = slit_min + slit_width
    mask = np.zeros(mask_shape)
    mask[:, 0:slit_min] = 1.0
    mask[:, slit_max:] = 1.0
    return mask


def select_slit_location(gt, min_slit_width, max_slit_width, min_observable=5):
    z_vals = tf.where(tf.reduce_sum(gt, axis=[0, 1, 3]))

    slit_width = tf.random.uniform(shape=[], minval=min_slit_width, maxval=max_slit_width, dtype=tf.int64)

    slit_min_possible = tf.reduce_min(z_vals) - slit_width + min_observable
    slit_max_possible = tf.reduce_max(z_vals) - min_observable
    slit_max_possible = tf.maximum(slit_max_possible, slit_min_possible + 1)

    slit_min = tf.random.uniform(shape=[],
                                 minval=slit_min_possible,
                                 maxval=slit_max_possible,
                                 dtype=tf.int64)

    return slit_min, slit_min + slit_width


def simulate_depth_image(vg):
    vg = conversions.format_voxelgrid(vg, False, False)
    size = vg.shape[1]
    z_inds = tf.expand_dims(tf.expand_dims(tf.range(size), axis=-1), axis=-1)
    z_inds = tf.repeat(tf.repeat(z_inds, size, axis=1), size, axis=2)
    z_inds = tf.cast(z_inds, tf.float32)
    dists = z_inds * vg + size * tf.cast(vg == 0, tf.float32)
    return tf.reduce_min(dists, axis=0)


@tf.function
def shift_voxelgrid(t, dx, dy, dz, pad_value, max_x, max_y, max_z):
    """
    Shifts a single (non-batched) voxelgrid of shape (x,y,z,channels)
    :param t: voxelgrid tensor to shift
    :param dx: x shift amount (tensor of shape [])
    :param dy: y shift amount
    :param dz: z shift amount
    :param pad_value: value to pad the new "empty" spaces
    :param max_x: max x shift
    :param max_y: max y shift
    :param max_z: max z shift
    :return:
    """
    a = np.abs(max_x)
    b = np.abs(max_y)
    c = np.abs(max_z)

    if a > 0:
        t = tf.pad(t, paddings=tf.constant([[a, a], [0, 0], [0, 0], [0, 0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, [dx], axis=[0])
        t = t[a:-a, :, :, :]

    if b > 0:
        t = tf.pad(t, paddings=tf.constant([[0, 0], [b, b], [0, 0], [0, 0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, [dy], axis=[1])
        t = t[:, b:-b, :, :]
    if c > 0:
        t = tf.pad(t, paddings=tf.constant([[0, 0], [0, 0], [c, c], [0, 0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, [dz], axis=[2])
        t = t[:, :, c:-c, :]

    return t


# @tf.function
def simulate_first_n_input(gt, n):
    gt_occ = gt
    gt_free = 1.0 - gt
    mask = tf.concat([tf.ones(n), tf.zeros(tf.size(gt) - n)], axis=0)
    shape = gt.shape
    gt_occ_masked = tf.reshape(tf.reshape(gt_occ, [-1]) * mask, shape)
    gt_free_masked = tf.reshape(tf.reshape(gt_free, [-1]) * mask, shape)
    return gt_occ_masked, gt_free_masked


# @tf.function
def simulate_first_random_input(gt):
    n = tf.random.uniform(shape=[1], minval=0, maxval=tf.size(gt), dtype=tf.int32)
    return simulate_first_n_input(gt, n)


def simulate_random_partial_completion_input(gt):
    gt_occ = gt
    gt_free = 1.0 - gt

    # mask_n = tf.random.uniform(shape=[1], minval=0, maxval=tf.size(gt), dtype=tf.int32)
    mask_n = tf.random.uniform(shape=[1])
    mask_n = tf.pow(mask_n, 8.0)
    mask_n = tf.cast(mask_n * tf.cast(tf.size(gt), tf.float32), tf.int32)
    mask = tf.concat([tf.ones(mask_n), tf.zeros(tf.size(gt) - mask_n)], axis=0)

    mask = tf.random.shuffle(mask)

    shape = gt.shape

    gt_occ_masked = tf.reshape(tf.reshape(gt_occ, [-1]) * mask, shape)
    gt_free_masked = tf.reshape(tf.reshape(gt_free, [-1]) * mask, shape)
    return gt_occ_masked, gt_free_masked


@memoize
def get_addressible_dataset(**kwargs):
    return AddressableDataset(**kwargs)


class AddressableDataset():
    def __init__(self, use_test=True, use_train=True, dataset_name="shapenet"):
        self.train_ds, self.test_ds = load_dataset(dataset_name=dataset_name,
                                                   metadata_only=True,
                                                   shuffle=False)
        self.train_map = {}
        self.test_map = {}
        self.train_names = []
        self.test_names = []

        if use_train:
            for i, elem in self.train_ds.enumerate():
                self.train_map[get_unique_name(elem)] = i
                self.train_names.append(get_unique_name(elem))
        if use_test:
            for i, elem in self.test_ds.enumerate():
                self.test_map[get_unique_name(elem)] = i
                self.test_names.append(get_unique_name(elem))

    def get(self, unique_name, params=None):
        if unique_name in self.train_map:
            ds = self.train_ds.skip(self.train_map[unique_name])
        elif unique_name in self.test_map:
            ds = self.test_ds.skip(self.test_map[unique_name])
        else:
            raise Exception("No element {} in dataset".format(unique_name))

        ds = load_voxelgrids(ds.take(1))
        if params is None:
            ds = simulate_input(ds, 0, 0, 0)
            return next(ds.__iter__())

        ds = preprocess_dataset(ds, params)
        return next(ds.__iter__())

    def get_metadata(self, unique_name):
        if unique_name in self.train_map:
            ds = self.train_ds.skip(self.train_map[unique_name])
        elif unique_name in self.test_map:
            ds = self.test_ds.skip(self.test_map[unique_name])
        else:
            raise Exception("No element {} in dataset".format(unique_name))
        return next(ds.__iter__())


def get_dataset_path(dataset_name):
    paths = {"shapenet": shapenet_storage.shapenet_load_path,
             "ycb": ycb_storage.ycb_load_path}
    return paths[dataset_name]


def load_dataset(dataset_name, metadata_only=True, shuffle=True):
    """
    @param shuffle: shuffle the dataset
    @param dataset_name: either "ycb" or "shapenet"
    @param metadata_only: if True, only loads metadata without voxelgrids
    """
    if dataset_name == 'shapenet':
        train_data, test_data = load_shapenet_metadata([
            shapenet_storage.shape_map["mug"]], shuffle=shuffle)
    elif dataset_name == 'ycb':
        train_data, test_data = load_ycb_metadata(shuffle=shuffle)
    else:
        raise Exception("Unknown dataset: {}".format(dataset_name))

    if not metadata_only:
        train_data = load_voxelgrids(train_data)
        test_data = load_voxelgrids(test_data)

    return train_data, test_data


# def load_shapenet(shapes="all", shuffle=True):
#     train_ds, test_ds = load_shapenet_metadata(shapes, shuffle)
#     return load_voxelgrids(train_ds), load_voxelgrids(test_ds)


def load_shapenet_metadata(shapes="all", shuffle=True):
    print("Loading Shapenet dataset")
    return _load_metadata_train_or_test(shapes, shuffle, "train"), \
           _load_metadata_train_or_test(shapes, shuffle, "test"),


def load_ycb_metadata(shuffle=True):
    print("Loading YCB dataset")
    return _load_metadata_train_or_test(shuffle=shuffle, prefix="train", record_path=ycb_storage.ycb_record_path), \
           _load_metadata_train_or_test(shuffle=shuffle, prefix="test", record_path=ycb_storage.ycb_record_path),


def preprocess_dataset(dataset, params):
    dataset = simulate_input(dataset,
                             params['translation_pixel_range_x'],
                             params['translation_pixel_range_y'],
                             params['translation_pixel_range_z'],
                             sim_input_fn=simulate_2_5D_input)

    if params['apply_slit_occlusion']:
        print("Applying slit occlusion")
        dataset = apply_slit_occlusion(dataset)

    if params['simulate_partial_completion']:
        dataset = simulate_partial_completion(dataset)
    if params['simulate_random_partial_completion']:
        dataset = simulate_random_partial_completion(dataset)
    return dataset


def preprocess_test_dataset(dataset, params):
    dataset = simulate_input(dataset, 0, 0, 0, sim_input_fn=simulate_2_5D_input)

    if params['apply_slit_occlusion']:
        print("Applying fixed slit occlusion")
        dataset = apply_fixed_slit_occlusion(dataset, params['slit_start'], params['slit_width'])
        dataset = apply_deterministic_slit_occlusion(dataset)

    return dataset


def _load_metadata_train_or_test(shapes="all", shuffle=True, prefix="train",
                                 record_path=shapenet_storage.shapenet_record_path):
    records = [f for f in record_path.iterdir()
               if f.name == prefix + "_filepaths.pkl"]
    if shapes != "all":
        print("Not yet handling partial loading")

    ds = None
    for fp in records:
        if ds:
            ds = ds.concatenate(read_metadata_from_filelist(fp, shuffle))
        else:
            ds = read_metadata_from_filelist(fp, shuffle)
    return ds


def read_metadata_from_filelist(record_file, shuffle):
    """
    Reads from a tfrecord file of paths and augmentations
    Loads the binvox files, simulates the input tensors, and returns a dataset
    """
    # print("Reading from filepath record")
    # raw_dataset = tf.data.TFRecordDataset(record_file.as_posix())
    #
    # keys = ['id', 'shape_category', 'fp', 'augmentation']
    # tfrecord_description = {k: tf.io.FixedLenFeature([], tf.string) for k in keys}
    #
    # def _parse_record_function(example_proto):
    #     # Parse the input tf.Example proto using the dictionary above.
    #     example = tf.io.parse_single_example(example_proto, tfrecord_description)
    #     # return pickle.loads(example_proto.numpy())
    #     return example
    #
    # if shuffle:
    #     raw_dataset = raw_dataset.shuffle(10000)
    #
    # parsed_dataset = raw_dataset.map(_parse_record_function)
    with open(record_file.as_posix()) as f:
        filelist = pickle.load(f)
    ds = tf.data.Dataset.from_tensor_slices(filelist)

    if shuffle:
        ds = ds.shuffle(10000)

    return ds


def load_voxelgrids(metadata_ds):
    def _get_shape(_raw_dataset):
        e = next(_raw_dataset.__iter__())
        return tf.TensorShape(e["shape"])

    shape = _get_shape(metadata_ds)

    def _load_voxelgrids(elem):
        fp = elem['filepath']
        gt = tf.numpy_function(load_gt_only, [fp], tf.float32)
        gt.set_shape(shape)
        elem['gt_occ'] = gt
        elem['gt_free'] = 1.0 - gt

        return elem

    return metadata_ds.map(_load_voxelgrids)


def shift_dataset_element(elem, x, y, z):
    """
    Shift the voxelgrid and bounding box of elem by a random amount, up to the limits [x,y,z]
    :param elem:
    :param x: maximum x shift
    :param y: maximum y shift
    :param z: maximum z shift
    :return:
    """
    dx = 0
    dy = 0
    dz = 0
    if x > 0:
        dx = tf.random.uniform(shape=[], minval=-x, maxval=x, dtype=tf.int64)
    if y > 0:
        dy = tf.random.uniform(shape=[], minval=-y, maxval=y, dtype=tf.int64)
    if z > 0:
        dz = tf.random.uniform(shape=[], minval=-z, maxval=z, dtype=tf.int64)
    elem['gt_occ'] = shift_voxelgrid(elem['gt_occ'], dx, dy, dz, 0.0, x, y, z)
    elem['gt_free'] = shift_voxelgrid(elem['gt_free'], dx, dy, dz, 1.0, x, y, z)
    elem['bounding_box'] += tf.cast([[dx, dy, dz]], tf.float64) * 0.01
    return elem


def shift_bounding_box_only(elem, x, y, z):
    """
    Shift only the bounding box of elem by a random amount, up to the limits [x,y,z]
    :param elem:
    :param x: maximum x shift
    :param y: maximum y shift
    :param z: maximum z shift
    :return:
    """
    dx = 0
    dy = 0
    dz = 0
    if x > 0:
        dx = tf.random.uniform(shape=[], minval=-x, maxval=x, dtype=tf.int64)
    if y > 0:
        dy = tf.random.uniform(shape=[], minval=-y, maxval=y, dtype=tf.int64)
    if z > 0:
        dz = tf.random.uniform(shape=[], minval=-z, maxval=z, dtype=tf.int64)
    elem['bounding_box'] += tf.cast([[dx, dy, dz]], tf.float64) * 0.01
    return elem


def simulate_input(dataset, x, y, z, sim_input_fn=simulate_2_5D_input):
    def _simulate_input(example):
        known_occ, known_free = tf.numpy_function(sim_input_fn, [example['gt_occ']],
                                                  [tf.float32, tf.float32])
        known_occ.set_shape(example['gt_occ'].shape)
        known_free.set_shape(example['gt_occ'].shape)
        example['known_occ'] = known_occ
        example['known_free'] = known_free
        return example

    def _shift(elem):
        return shift_dataset_element(elem, x, y, z)

    return dataset.map(_shift, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(_simulate_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def apply_slit_occlusion(dataset):
    def _apply_slit_occlusion(elem):
        slit_min, slit_max = select_slit_location(elem['gt_occ'], min_slit_width=5, max_slit_width=30,
                                                  min_observable=5)
        ko, kf = tf.numpy_function(simulate_slit_occlusion, [elem['known_occ'], elem['known_free'],
                                                             slit_min, slit_max], [tf.float32, tf.float32])

        # ko, kf = simulate_slit_occlusion(elem['known_occ'].numpy(), elem_raw['known_free'].numpy(),
        #                              slitmin, slitmax)
        elem['known_occ'] = ko
        elem['known_free'] = kf
        return elem

    return dataset.map(_apply_slit_occlusion)


def apply_deterministic_slit_occlusion(dataset):
    #Deprecated
    raise Exception("Deprecated")

    # def _apply_slit_occlusion(elem):
    #
    #     slit_width=6
    #
    #     z_vals = tf.where(tf.reduce_sum(elem['gt_occ'], axis=[0, 1, 3]))
    #     slit_min = tf.reduce_min(z_vals) + 2
    #     slit_max = slit_min + slit_width
    #
    #
    #     ko, kf = tf.numpy_function(simulate_slit_occlusion, [elem['known_occ'], elem['known_free'],
    #                                                          slit_min, slit_max], [tf.float32, tf.float32])
    #
    #     # ko, kf = simulate_slit_occlusion(elem['known_occ'].numpy(), elem_raw['known_free'].numpy(),
    #     #                              slitmin, slitmax)
    #     elem['known_occ'] = ko
    #     elem['known_free'] = kf
    #     return elem
    #
    # return dataset.map(_apply_slit_occlusion)
    return apply_fixed_slit_occlusion(dataset, 28, 6)


def apply_fixed_slit_occlusion(dataset, slit_min, slit_width):
    def _apply_slit_occlusion(elem):

        ko, kf = tf.numpy_function(simulate_slit_occlusion, [elem['known_occ'], elem['known_free'],
                                                             slit_min, slit_min+slit_width], [tf.float32, tf.float32])
        elem['known_occ'] = ko
        elem['known_free'] = kf
        return elem

    return dataset.map(_apply_slit_occlusion)


def simulate_partial_completion(dataset):
    def _add_partial_gt(elem):
        partial_occ, partial_free = simulate_first_random_input(elem['gt_occ'])
        elem['known_occ'] = tf.clip_by_value(elem['known_occ'] + partial_occ, 0.0, 1.0)
        elem['known_free'] = tf.clip_by_value(elem['known_free'] + partial_free, 0.0, 1.0)
        return elem

    return dataset.map(_add_partial_gt)


def simulate_random_partial_completion(dataset):
    def _add_partial_gt(elem):
        partial_occ, partial_free = simulate_random_partial_completion_input(elem['gt_occ'])
        elem['known_occ'] = tf.clip_by_value(elem['known_occ'] + partial_occ, 0.0, 1.0)
        elem['known_free'] = tf.clip_by_value(elem['known_free'] + partial_free, 0.0, 1.0)
        return elem

    return dataset.map(_add_partial_gt)


def simulate_condition_occ(dataset, turn_on_prob=0.0, turn_off_prob=0.0):
    def _add_conditional(elem):
        x = elem['gt_occ']
        x = x + tf.cast(tf.random.uniform(x.shape) < turn_on_prob, tf.float32)
        x = x - tf.cast(tf.random.uniform(x.shape) < turn_off_prob, tf.float32)
        x = tf.clip_by_value(x, 0.0, 1.0)
        elem['conditioned_occ'] = x
        return elem

    return dataset.map(_add_conditional)


def get_unique_name(datum):
    """
    Returns a unique name for the datum
    @param datum:
    @return:
    """
    return datum['id'].numpy() + datum['augmentation'].numpy()


if __name__ == "__main__":
    print("Not meant to be executed as main script")
