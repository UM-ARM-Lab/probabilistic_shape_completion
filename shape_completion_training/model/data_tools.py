#! /usr/bin/env python
from __future__ import print_function

import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from shape_completion_training import binvox_rw
import numpy as np
import sys

import IPython



shape_map = {"airplane":"02691156",
             "mug":"03797390"}



cur_path = os.path.dirname(__file__)
shapenet_load_path = join(cur_path, "../data/ShapeNetCore.v2_augmented")
shapenet_record_path = join(cur_path, "../data/ShapeNetCore.v2_augmented/tfrecords/")
# shapenet_record_path = join(cur_path, "../data/ShapeNetCore.v2/tfrecords/")

obj = "025_mug"
base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/"


gt_path = base_path + obj + "/gt/"
occ_path = base_path + obj + "/train_x_occ/"
non_occ_path = base_path + obj + "/non_occupy/"

record_name = "./tmp_data/" + obj + ".tfrecord"



def shapenet_labels(human_names):
    return [shape_map[hn] for hn in human_names]



"""Loads data from file. From a TF_Record, if asked"""
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



def get_simulated_input(gt):
    known_free = []
    known_occ = []

    known_occ = gt
    known_free = 1.0 - gt
    
    return {"known_free": known_free, "known_occ": known_occ, "gt":gt}


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



"""
Processes shapenet files and saves necessary data in tfrecords

params:
shape_ids: a list of strings of shapenet object categories, or "all"
"""
def write_shapenet_to_tfrecord(shape_ids = "all"):
    if shape_ids == "all":
        
        shape_ids = [f for f in os.listdir(shapenet_load_path)
                     if os.path.isdir(join(shapenet_load_path, f))]
        shape_ids.sort()


    for i in range(0, len(shape_ids)): #Replase with iteration over folders
        shape_id = shape_ids[i]
        shape_path = join(shapenet_load_path, shape_id)
        gt = []
        data = {'id':[], 'shape_category':[]}

        print("")
        print("{}/{}: Loading {}. ({} models)".format(i+1, len(shape_ids), shape_id, len(os.listdir(shape_path))))
        objs = os.listdir(shape_path)
        for i, obj in zip(range(len(objs)), objs):
            obj_fp = join(shape_path, obj, "models")
            print("    {}/{} Processing {}".format(i+1, len(objs), obj), end="")
            sys.stdout.flush()

            
            augs = [f[len('model_augmented_'):].split('.')[0]
                    for f in os.listdir(obj_fp)
                    if f.startswith("model_augmented")
                    if f.endswith(".binvox")]
            augs = list(set(augs))
            augs.sort()
            for augmentation in augs:
                gt_vox = load_gt_voxels(obj_fp, augmentation)
                
                gt.append(gt_vox)
                data['shape_category'].append(shape_id)
                data['id'].append(obj)
            sys.stdout.write('\033[2K\033[1G')

                    
                # gt_vox = np.array(gt_vox, dtype=np.float32)
        gt = np.array(gt, dtype=np.float32)
        gt = np.expand_dims(gt, axis=4)
        data.update(get_simulated_input(gt))

        ds = tf.data.Dataset.from_tensor_slices(data)
        # IPython.embed()
        
        print("      Writing {}".format(shape_id))
        sys.stdout.flush()
        write_to_tfrecord(ds, join(shapenet_record_path, shape_id + ".tfrecord"))

def load_shapenet(shapes = "all"):
    if shapes == "all":
        shapes = [f[:-9] for f in os.listdir(shapenet_record_path)
                  if f.endswith(".tfrecord")]
        shapes.sort()


    ds = None
    for shape in shapes:

        fp = join(shapenet_record_path, shape + ".tfrecord")
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
                'gt': _bytes_feature(tf.io.serialize_tensor(elem['gt']).numpy()),
                'known_occ': _bytes_feature(tf.io.serialize_tensor(elem['known_occ']).numpy()),
                'known_free': _bytes_feature(tf.io.serialize_tensor(elem['known_free']).numpy()),
                'shape_category': _bytes_feature(elem['shape_category'].numpy()),
                'id': _bytes_feature(elem['id'].numpy()),
                }
            # IPython.embed()
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())
            # writer.write(tf.io.serialize_tensor(elem[0]).numpy())
            

            

def read_from_record(record_file):
    raw_dataset = tf.data.TFRecordDataset(record_file)

    voxelgrid_description = {
        'gt': tf.io.FixedLenFeature([], tf.string),
        'known_occ': tf.io.FixedLenFeature([], tf.string),
        'known_free': tf.io.FixedLenFeature([], tf.string),
        'shape_category': tf.io.FixedLenFeature([], tf.string, default_value=''),
        'id': tf.io.FixedLenFeature([], tf.string, default_value=''),
    }


    def _get_shape(_raw_dataset):
        e = next(_raw_dataset.__iter__())
        example = tf.io.parse_single_example(e, voxelgrid_description)
        t = tf.io.parse_tensor(example['gt'], tf.float32)
        return t.shape
    # IPython.embed()
    shape = _get_shape(raw_dataset)
        

    def _parse_voxelgrid_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.io.parse_single_example(example_proto, voxelgrid_description)
        gt = tf.io.parse_tensor(example['gt'], tf.float32)
        known_occ = tf.io.parse_tensor(example['known_occ'], tf.float32)
        known_free = tf.io.parse_tensor(example['known_free'], tf.float32)
        category = example['shape_category']
        id = example['id']
        gt.set_shape(shape)
        known_occ.set_shape(shape)
        known_free.set_shape(shape)
        # return ({'gt': gt, 'known_occ': known_occ, 'known_free': known_free,
        #          'shape_category': category, 'id': id},
        #         gt)
        return {'gt': gt, 'known_occ': known_occ, 'known_free': known_free,
                 'shape_category': category, 'id': id}
        # return (known_occ, gt)


    parsed_dataset = raw_dataset.map(_parse_voxelgrid_function)

    # elem = next(parsed_dataset.__iter__())
    # IPython.embed()

    
    return parsed_dataset




if __name__=="__main__":
    print("Not meant to be executed as main script")
    
