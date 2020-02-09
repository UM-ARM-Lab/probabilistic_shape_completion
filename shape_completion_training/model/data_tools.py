#! /usr/bin/env python
from __future__ import print_function

import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from shape_completion_training import binvox_rw
import numpy as np
import sys
import progressbar

import IPython



shape_map = {"airplane":"02691156",
             "mug":"03797390"}



cur_path = os.path.dirname(__file__)
shapenet_load_path = join(cur_path, "../data/ShapeNetCore.v2_augmented")
shapenet_record_path = join(cur_path, "../data/ShapeNetCore.v2_augmented/tfrecords/gt")
# shapenet_record_path = join(cur_path, "../data/ShapeNetCore.v2_augmented/tfrecords/2.5D")





def shapenet_labels(human_names):
    return [shape_map[hn] for hn in human_names]


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


def get_gt_simulated_input(gt):
    gt_occ = gt
    gt_free = 1.0 - gt
    known_occ = gt
    known_free = gt_free
    
    return {"known_free": known_free, "known_occ": known_occ, "gt_occ":gt_occ, "gt_free":gt_free}

def get_2_5D_simulated_input(gt):
    gt_occ = gt
    gt_free = 1.0 - gt

    IPython.embed()
    
    known_occ = gt
    known_free = gt_free
    
    return {"known_free": known_free, "known_occ": known_occ, "gt_occ":gt_occ, "gt_free":gt_free}


def load_gt_voxels(filepath, augmentation):


    binvox_wire_fp = join(filepath, 'model_augmented_' + augmentation + '.wire.binvox')    
    with open(binvox_wire_fp) as f:
        wire_vox = binvox_rw.read_as_3d_array(f).data

    binvox_mesh_fp = join(filepath, 'model_augmented_' + augmentation + '.mesh.binvox')    
    with open(binvox_mesh_fp) as f:
        mesh_vox = binvox_rw.read_as_3d_array(f).data

    # cuda_binvox_fp = join(filepath, 'model_augmented_' + augmentation + '.obj_64.binvox')
    # with open(cuda_binvox_fp) as f:
    #     cuda_gt_vox = binvox_rw.read_as_3d_array(f).data

    combined = wire_vox * 1.0 + mesh_vox * 1.0
    return np.clip(combined, 0, 1)


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

    
    for i in range(0, len(shape_ids)): #Replace with iteration over folders
        category = shape_ids[i]
        shape_path = join(shapenet_load_path, category)
        objs = os.listdir(shape_path)
        for obj in objs:
            obj_fp = join(shape_path, obj, "models")
            augs = [f[len('model_augmented_'):].split('.')[0]
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

"""
Groups a single list of ShapenetRecords into groups of lists. 
Each group contains only one category and is at most group_size long
"""
def group_shapenet_files(shapenet_files, group_size):
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
    

"""
Processes shapenet files and saves necessary data in tfrecords

params:
shape_ids: a list of strings of shapenet object categories, or "all"
"""
def write_shapenet_to_tfrecord(shape_ids = "all"):
    all_files = get_all_shapenet_files(shape_ids)
    widgets = [
        ' ', progressbar.Counter(), '/', str(len(all_files)),
        ' ', progressbar.Variable("message"), ' ',
        progressbar.Bar(),
        ' [', progressbar.Timer(), '] '
        ]

    i = 0
    with progressbar.ProgressBar(widgets=widgets, max_value=len(all_files)) as bar:
        for group_name, group in group_shapenet_files(all_files, 1000):
            data = {'id':[], 'shape_category':[]}
            gt = []
            for sr in group:
                i+=1
                bar.update(i, message=group_name)
                
                gt_vox = load_gt_voxels(sr.filepath, sr.augmentation)
                gt.append(gt_vox)
            
                data['id'].append(sr.id)
                data['shape_category'].append(sr.category)

            bar.update(i, message="Creating tensor")
            gt = np.array(gt, dtype=np.float32)
            gt = np.expand_dims(gt, axis=4)
            data.update(get_gt_simulated_input(gt))
            ds = tf.data.Dataset.from_tensor_slices(data)

            bar.update(i, message="Writing to tf_record")
            write_to_tfrecord(ds, join(shapenet_record_path, group_name + ".tfrecord"))

        # gt = []
        # data = {'id':[], 'shape_category':[]}

        # print("")
        # print("{}/{}: Loading {}. ({} models)".format(i+1, len(shape_ids), shape_id, len(os.listdir(shape_path))))

        #     print("    {}/{} Processing {}".format(i+1, len(objs), obj), end="")
        #     sys.stdout.flush()

            
        #         gt_vox = load_gt_voxels(obj_fp, augmentation)
                
        #         gt.append(gt_vox)
        #         data['shape_category'].append(shape_id)
        #         data['id'].append(obj)
        #     sys.stdout.write('\033[2K\033[1G')

                    
        #         # gt_vox = np.array(gt_vox, dtype=np.float32)
        # gt = np.array(gt, dtype=np.float32)
        # gt = np.expand_dims(gt, axis=4)
        # data.update(get_simulated_input(gt))

        # ds = tf.data.Dataset.from_tensor_slices(data)
        # # IPython.embed()
        
        # print("      Writing {}".format(shape_id))
        # sys.stdout.flush()
        # write_to_tfrecord(ds, join(shapenet_record_path, shape_id + ".tfrecord"))

        
def load_shapenet(shapes = "all"):
    records = [f for f in os.listdir(shapenet_record_path)
               if f.endswith(".tfrecord")]
    if shapes != "all":
        print("Not yet handling partial loading")

    ds = None
    for fp in [join(shapenet_record_path, r) for r in records]:
        if ds:
            ds = ds.concatenate(read_from_record(fp))
        else:
            ds = read_from_record(fp)
    return ds



def write_to_tfrecord(dataset, record_file):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    """Returns a float_list from a float / double."""
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    with tf.io.TFRecordWriter(record_file) as writer:
        for elem in dataset:
            feature={
                'gt_occ': _bytes_feature(tf.io.serialize_tensor(elem['gt_occ']).numpy()),
                'gt_free': _bytes_feature(tf.io.serialize_tensor(elem['gt_free']).numpy()),
                'known_occ': _bytes_feature(tf.io.serialize_tensor(elem['known_occ']).numpy()),
                'known_free': _bytes_feature(tf.io.serialize_tensor(elem['known_free']).numpy()),
                'shape_category': _bytes_feature(elem['shape_category'].numpy()),
                'id': _bytes_feature(elem['id'].numpy()),
                }
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())
            

            

def read_from_record(record_file):
    raw_dataset = tf.data.TFRecordDataset(record_file)

    voxelgrid_description = {
        'gt_occ': tf.io.FixedLenFeature([], tf.string),
        'gt_free': tf.io.FixedLenFeature([], tf.string),
        'known_occ': tf.io.FixedLenFeature([], tf.string),
        'known_free': tf.io.FixedLenFeature([], tf.string),
        'shape_category': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }


    def _get_shape(_raw_dataset):
        e = next(_raw_dataset.__iter__())
        example = tf.io.parse_single_example(e, voxelgrid_description)
        t = tf.io.parse_tensor(example['gt_occ'], tf.float32)
        return t.shape
    shape = _get_shape(raw_dataset)
        

    def _parse_voxelgrid_function(example_proto):
        voxel_grid_fields = ['gt_occ', 'gt_free', 'known_occ', 'known_free']
        
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.io.parse_single_example(example_proto, voxelgrid_description)
        for field in voxel_grid_fields:
            example[field] = tf.io.parse_tensor(example[field], tf.float32)
            example[field].set_shape(shape)

        return example
        
    parsed_dataset = raw_dataset.map(_parse_voxelgrid_function)
    return parsed_dataset



# obj = "025_mug"
# base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/"


# gt_path = base_path + obj + "/gt/"
# occ_path = base_path + obj + "/train_x_occ/"
# non_occ_path = base_path + obj + "/non_occupy/"

# record_name = "./tmp_data/" + obj + ".tfrecord"

"""
Deprecated! Use load_shapent instead
Loads data from file. From a TF_Record, if asked
"""

def load_data(from_record=False):



    if from_record:
        return read_from_record(record_name)
    

    files = [f for f in os.listdir(occ_path)]
    files.sort()

    data = []

    for filename in files:
        prefix = filename.split("occupy")[0]
        print(prefix)

        with open(join(gt_path,prefix + "gt.binvox")) as f:
            gt_vox = binvox_rw.read_as_3d_array(f).data

        data.append(gt_vox)

    data = np.array(data)
    data = np.expand_dims(data, axis=4)
    # IPython.embed()
    ds = tf.data.Dataset.from_tensor_slices((data, data))
    write_to_tfrecord(ds, record_name)
    return ds


def maxpool_np(array3d, scale):
    a, b, c = array3d.shape
    s = scale
    return array3d.reshape(a/s, s, b/s, s, c/s, s).max(axis=(1,3,5))


if __name__=="__main__":
    print("Not meant to be executed as main script")
    
