#! /usr/bin/env python
from __future__ import print_function

import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from shape_completion_training import binvox_rw
import numpy as np

shape_map = {"airplane":"02691156",
             "mug":"03797390"}

cur_path = os.path.dirname(__file__)
shapenet_load_path = join(cur_path, "../../../data/ShapeNetCore.v2_augmented")
# shapenet_record_path = join(cur_path, "../data/ShapeNetCore.v2_augmented/tfrecords/gt")
# shapenet_record_path = join(cur_path, "../data/ShapeNetCore.v2_augmented/tfrecords/2.5D")
shapenet_record_path = join(cur_path, "../../../data/ShapeNetCore.v2_augmented/tfrecords/filepath")


def shapenet_labels(human_names):
    return [shape_map[hn] for hn in human_names]


def simulate_omniscient_input(gt):
    """
    Given a single ground truth mask occupied list, return ground truth occupied and free, as well as simulate the known occupied and free
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
        unknown_mask = unknown_mask + gt_occ[h,:,:,0]
        unknown_mask = np.clip(unknown_mask, 0, 1)
    return known_occ, known_free


def shift_elem(elem, x, y, z):
    keys = ['gt_occ', 'known_occ', 'gt_free', 'known_free']
    pad_values = {'gt_occ':0.0,
                  'known_occ':0.0,
                  'gt_free':1.0,
                  'known_free':1.0}
    for k in keys:
        elem[k] = shift_tensor(elem[k], x, y, z, pad_values[k])
    return elem


@tf.function
def shift_tensor(t, dx, dy, dz, pad_value, max_x, max_y, max_z):
    # p = np.madx(np.abs([dx,dy,dz]))
    a = np.abs(max_x)
    b = np.abs(max_y)
    c = np.abs(max_z)

    if a > 0:
        t = tf.pad(t, paddings=tf.constant([[a,a], [0,0], [0,0], [0,0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, dx, axis=[0])
        t = t[a:-a, :, :, :]

    if b > 0:
        t = tf.pad(t, paddings=tf.constant([[0,0], [b,b], [0,0], [0,0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, dy, axis=[1])
        t = t[:, b:-b, :, :]
    if c > 0:
        t = tf.pad(t, paddings=tf.constant([[0,0], [0,0], [c,c],[0,0]]),
                   mode="CONSTANT", constant_values=pad_value)
        t = tf.roll(t, dz, axis=[2])
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


def load_gt_voxels(filepath, augmentation):
    """
    Loads ground truth voxels into a np.array

    filepath: string filepath to the "models" folder for this shape
    augmentation: string identifying the augmentation
    """
    binvox_wire_fp = join(filepath, 'model_augmented_' + augmentation + '.wire.binvox')    
    with open(binvox_wire_fp) as f:
        wire_vox = binvox_rw.read_as_3d_array(f).data

    binvox_mesh_fp = join(filepath, 'model_augmented_' + augmentation + '.mesh.binvox')    
    with open(binvox_mesh_fp) as f:
        mesh_vox = binvox_rw.read_as_3d_array(f).data

    # cuda_binvox_fp = join(filepath, 'model_augmented_' + augmentation + '.obj_64.binvox')
    # with open(cuda_binvox_fp) as f:
    #     cuda_gt_vox = binvox_rw.read_as_3d_array(f).data

    gt = wire_vox * 1.0 + mesh_vox * 1.0
    gt = np.clip(gt, 0, 1)
    gt = np.array(gt, dtype=np.float32)
    gt = np.expand_dims(gt, axis=4)
    return gt


class ShapenetRecord:
    def __init__(self):
        self.filepath = None
        self.category = None
        self.id = None
        self.augmentation = None


def get_all_shapenet_files(shape_ids):
    shapenet_files = []
    if shape_ids == "all":
        
        shape_ids = [f for f in os.listdir(shapenet_load_path)
                     if os.path.isdir(join(shapenet_load_path, f))]
        shape_ids.sort()

    for i in range(0, len(shape_ids)):
        category = shape_ids[i]
        shape_path = join(shapenet_load_path, category)
        objs = os.listdir(shape_path)
        for obj in objs:
            obj_fp = join(shape_path, obj, "models")
            augs = [f[len('model_augmented_'):-len('.wire.binvox')]
                    for f in os.listdir(obj_fp)
                    if f.startswith("model_augmented")
                    if f.endswith(".binvox")]

            augs = list(set(augs))
            augs.sort()
            for augmentation in augs:
                sr = ShapenetRecord()
                sr.id = obj
                sr.filepath = obj_fp
                sr.category = category
                sr.augmentation = augmentation
                shapenet_files.append(sr)
    return shapenet_files


def group_shapenet_files(shapenet_files, group_size):
    """
    Groups a single list of ShapenetRecords into groups of lists.
    Each group contains only one category and is at most group_size long
    """
    groups = []
    group = []
    category = shapenet_files[0].category
    group_num = 1

    def get_group_name(category, num):
        return "{}_{:02d}".format(category, num)
    
    i = 0
    for sr in shapenet_files:
        if sr.category != category:
            group_num = 1
        if sr.category != category or i >= group_size:
            groups.append((get_group_name(category, group_num), group))
            group = []
            i = 0
            category = sr.category
            group_num += 1
        i += 1
        group.append(sr)
    groups.append((get_group_name(category, group_num), group))
    return groups


def load_shapenet(shapes = "all", shuffle=True):
    ds = load_shapenet_metadata(shapes, shuffle)
    return load_voxelgrids(ds)


def load_shapenet_metadata(shapes = "all", shuffle=True):
    records = [f for f in os.listdir(shapenet_record_path)
               if f.endswith(".tfrecord")]
    if shapes != "all":
        print("Not yet handling partial loading")

    ds = None
    for fp in [join(shapenet_record_path, r) for r in records]:
        if ds:
            ds = ds.concatenate(read_metadata_from_tfrecord(fp, shuffle))
        else:
            ds = read_metadata_from_tfrecord(fp, shuffle)
    return ds


def write_shapenet_to_tfrecord(shape_ids = "all"):
    all_files = get_all_shapenet_files(shape_ids)
    data = {'id':[], 'shape_category':[], 'fp':[], 'augmentation':[]}
    for sr in all_files:
        data['id'].append(sr.id)
        data['shape_category'].append(sr.category)
        data['fp'].append(sr.filepath)
        data['augmentation'].append(sr.augmentation)

    ds = tf.data.Dataset.from_tensor_slices(data)
    write_to_tfrecord(ds, join(shapenet_record_path, "filepaths.tfrecord"))


def write_to_tfrecord(dataset, record_file):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    with tf.io.TFRecordWriter(record_file) as writer:
        for elem in dataset:
            feature={k: _bytes_feature(elem[k].numpy()) for k in elem}
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())


def read_metadata_from_tfrecord(record_file, shuffle):
    """
    Reads from a tfrecord file of paths and augmentations
    Loads the binvox files, simulates the input tensors, and returns a dataset
    """
    print("Reading from filepath record")
    raw_dataset = tf.data.TFRecordDataset(record_file)

    keys = ['id', 'shape_category', 'fp', 'augmentation']
    tfrecord_description = {k: tf.io.FixedLenFeature([], tf.string) for k in keys}

    def _parse_record_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.io.parse_single_example(example_proto, tfrecord_description)
        return example

    cache_name = "ds.cache"
    if shuffle:
        raw_dataset = raw_dataset.shuffle(10000)
        cache_name = "shuffled_ds.cache"
    cache_fp = join(shapenet_record_path, cache_name)
        
    parsed_dataset = raw_dataset.map(_parse_record_function)
    # parsed_dataset = parsed_dataset.cache(cache_fp)
    return parsed_dataset


def load_voxelgrids(metadata_ds):
    def _get_shape(_raw_dataset):
        e = next(_raw_dataset.__iter__())
        gt = tf.numpy_function(load_gt_voxels, [e['fp'], e['augmentation']], tf.float32)
        return gt.shape
    shape = _get_shape(metadata_ds)

    def _load_voxelgrids(elem):
        aug = elem['augmentation']
        fp = elem['fp']
        gt = tf.numpy_function(load_gt_voxels, [fp, aug], tf.float32)
        gt.set_shape(shape)
        elem['gt_occ'] = gt
        elem['gt_free'] = 1.0-gt
        # gt_occ.set_shape(shape)
        # gt_free.set_shape(shape)
        return elem
    return metadata_ds.map(_load_voxelgrids)


def simulate_input(dataset, x, y, z, sim_input_fn=simulate_2_5D_input):
    def _simulate_input(example):
        known_occ, known_free = tf.numpy_function(sim_input_fn, [example['gt_occ']],
                                                  [tf.float32, tf.float32])
        known_occ.set_shape(example['gt_occ'].shape)
        known_free.set_shape(example['gt_occ'].shape)
        example['known_occ'] = known_occ
        example['known_free'] = known_free
        return example

    def _shift(example):
        dx = 0
        dy = 0
        dz = 0
        if x > 0:
            dx = tf.random.uniform(shape=[1], minval = -x, maxval=x, dtype=tf.int64)
        if y > 0:
            dy = tf.random.uniform(shape=[1], minval = -x, maxval=x, dtype=tf.int64)
        if z > 0:
            dz = tf.random.uniform(shape=[1], minval = -x, maxval=x, dtype=tf.int64)
        example['gt_occ'] = shift_tensor(example['gt_occ'], dx, dy, dz, 0.0, x, y, z)
        example['gt_free'] = shift_tensor(example['gt_free'], dx, dy, dz, 1.0, x, y, z)
        return example

    return dataset.map(_shift, num_parallel_calls=tf.data.experimental.AUTOTUNE)\
                  .map(_simulate_input, num_parallel_calls=tf.data.experimental.AUTOTUNE)


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


def simulate_condition_occ(dataset, turn_on_prob = 0.0, turn_off_prob=0.0):
    def _add_conditional(elem):
        x = elem['gt_occ']
        x = x + tf.cast(tf.random.uniform(x.shape) < turn_on_prob, tf.float32)
        x = x - tf.cast(tf.random.uniform(x.shape) < turn_off_prob, tf.float32)
        x = tf.clip_by_value(x, 0.0, 1.0)
        elem['conditioned_occ'] = x
        return elem

    return dataset.map(_add_conditional)


def add_angle(dataset):
    def _augmentation_to_angle(augmentation):
        return np.float32(augmentation.split("_")[-1])

    def _extract_angle(elem):
        angle = tf.numpy_function(_augmentation_to_angle, [elem['augmentation']], tf.float32)
        elem['angle'] = angle
        return elem

    return dataset.map(_extract_angle)


###################################################################
##   OLD WAY OF TFRECORDS
###################################################################


# """
# Processes shapenet files and saves necessary data in tfrecords

# params:
# shape_ids: a list of strings of shapenet object categories, or "all"
# """
# def write_shapenet_to_tfrecord(shape_ids = "all"):
#     all_files = get_all_shapenet_files(shape_ids)
#     widgets = [
#         ' ', progressbar.Counter(), '/', str(len(all_files)),
#         ' ', progressbar.Variable("message"), ' ',
#         progressbar.Bar(),
#         ' [', progressbar.Timer(), '] '
#         ]

#     i = 0
#     with progressbar.ProgressBar(widgets=widgets, max_value=len(all_files)) as bar:
#         for group_name, group in group_shapenet_files(all_files, 1000):
#             data = {'id':[], 'shape_category':[]}
#             gt = []
#             for sr in group:
#                 i+=1
#                 bar.update(i, message=group_name)
                
#                 gt_vox = load_gt_voxels(sr.filepath, sr.augmentation)
#                 gt.append(gt_vox)
            
#                 data['id'].append(sr.id)
#                 data['shape_category'].append(sr.category)

#             bar.update(i, message="Creating tensor")
#             gt = np.array(gt, dtype=np.float32)
#             gt = np.expand_dims(gt, axis=4)
#             data.update(get_2_5D_simulated_input(gt))
#             ds = tf.data.Dataset.from_tensor_slices(data)

#             bar.update(i, message="Writing to tf_record")
#             write_to_tfrecord(ds, join(shapenet_record_path, group_name + ".tfrecord"))






        
# def load_shapenet(shapes = "all"):
#     records = [f for f in os.listdir(shapenet_record_path)
#                if f.endswith(".tfrecord")]
#     if shapes != "all":
#         print("Not yet handling partial loading")

#     ds = None
#     for fp in [join(shapenet_record_path, r) for r in records]:
#         if ds:
#             ds = ds.concatenate(read_from_record(fp))
#         else:
#             ds = read_from_record(fp)
#     return ds



# def write_to_tfrecord(dataset, record_file):
#     def _bytes_feature(value):
#         return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
#     """Returns a float_list from a float / double."""
#     def _float_feature(value):
#         return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

#     with tf.io.TFRecordWriter(record_file) as writer:
#         for elem in dataset:
#             feature={
#                 'gt_occ': _bytes_feature(tf.io.serialize_tensor(elem['gt_occ']).numpy()),
#                 'gt_free': _bytes_feature(tf.io.serialize_tensor(elem['gt_free']).numpy()),
#                 'known_occ': _bytes_feature(tf.io.serialize_tensor(elem['known_occ']).numpy()),
#                 'known_free': _bytes_feature(tf.io.serialize_tensor(elem['known_free']).numpy()),
#                 'shape_category': _bytes_feature(elem['shape_category'].numpy()),
#                 'id': _bytes_feature(elem['id'].numpy()),
#                 }
#             features = tf.train.Features(feature=feature)
#             example = tf.train.Example(features=features)

#             writer.write(example.SerializeToString())
            

            

# def read_from_record(record_file):
#     raw_dataset = tf.data.TFRecordDataset(record_file)

#     voxelgrid_description = {
#         'gt_occ': tf.io.FixedLenFeature([], tf.string),
#         'gt_free': tf.io.FixedLenFeature([], tf.string),
#         'known_occ': tf.io.FixedLenFeature([], tf.string),
#         'known_free': tf.io.FixedLenFeature([], tf.string),
#         'shape_category': tf.io.FixedLenFeature([], tf.string, default_value=''),
#         'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     }


#     def _get_shape(_raw_dataset):
#         e = next(_raw_dataset.__iter__())
#         example = tf.io.parse_single_example(e, voxelgrid_description)
#         t = tf.io.parse_tensor(example['gt_occ'], tf.float32)
#         return t.shape
#     shape = _get_shape(raw_dataset)
        

#     def _parse_voxelgrid_function(example_proto):
#         voxel_grid_fields = ['gt_occ', 'gt_free', 'known_occ', 'known_free']
        
#         # Parse the input tf.Example proto using the dictionary above.
#         example = tf.io.parse_single_example(example_proto, voxelgrid_description)
#         for field in voxel_grid_fields:
#             example[field] = tf.io.parse_tensor(example[field], tf.float32)
#             example[field].set_shape(shape)

#         return example
        
#     parsed_dataset = raw_dataset.map(_parse_voxelgrid_function)
#     return parsed_dataset



# obj = "025_mug"
# base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/"


# gt_path = base_path + obj + "/gt/"
# occ_path = base_path + obj + "/train_x_occ/"
# non_occ_path = base_path + obj + "/non_occupy/"

# record_name = "./tmp_data/" + obj + ".tfrecord"

# """
# Deprecated! Use load_shapent instead
# Loads data from file. From a TF_Record, if asked
# """

# def load_data(from_record=False):



#     if from_record:
#         return read_from_record(record_name)
    

#     files = [f for f in os.listdir(occ_path)]
#     files.sort()

#     data = []

#     for filename in files:
#         prefix = filename.split("occupy")[0]
#         print(prefix)

#         with open(join(gt_path,prefix + "gt.binvox")) as f:
#             gt_vox = binvox_rw.read_as_3d_array(f).data

#         data.append(gt_vox)

#     data = np.array(data)
#     data = np.expand_dims(data, axis=4)
#     # IPython.embed()
#     ds = tf.data.Dataset.from_tensor_slices((data, data))
#     write_to_tfrecord(ds, record_name)
#     return ds


# def maxpool_np(array3d, scale):
#     a, b, c = array3d.shape
#     s = scale
#     return array3d.reshape(a/s, s, b/s, s, c/s, s).max(axis=(1,3,5))


if __name__=="__main__":
    print("Not meant to be executed as main script")
    
