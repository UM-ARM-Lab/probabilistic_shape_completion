#! /usr/bin/env python
import os
from os.path import join
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from shape_completion_training import binvox_rw
import numpy as np
import IPython


cur_path = os.path.dirname(__file__)
shapenet_load_path = join(cur_path, "../data/ShapeNetCore.v2")
shapenet_record_path = join(cur_path, "../data/ShapeNetCore.v2_tfrecords/")

obj = "025_mug"
base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/"


gt_path = base_path + obj + "/gt/"
occ_path = base_path + obj + "/train_x_occ/"
non_occ_path = base_path + obj + "/non_occupy/"

record_name = "./tmp_data/" + obj + ".tfrecord"






"Loads data from file. From a TF_Record, if asked"
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

def write_shapenet_to_tfrecord():
    shape_ids = [f for f in os.listdir(shapenet_load_path)
                 if os.path.isdir(join(shapenet_load_path, f))]
    shape_ids.sort()


    for i in range(18, len(shape_ids)): #Replase with iteration over folders
        shape_id = shape_ids[i]

        print()
        print("{}/{}: Loading {}".format(i, len(shape_ids), shape_id))
        
        fdr = join(shapenet_load_path, shape_id)

        data = []
        
        for obj in os.listdir(fdr):
            obj_fp = join(fdr, obj, "models", "model_normalized.solid.binvox")

            if not os.path.isfile(obj_fp):
                print("File not found: {}".format(obj_fp))
                print("Skipping")
                continue
            
            with open(obj_fp) as f:
                gt_vox = binvox_rw.read_as_3d_array(f).data
            
            data.append(gt_vox)

        data = np.array(data)
        data = np.expand_dims(data, axis=4)

        ds = tf.data.Dataset.from_tensor_slices((data, data))
        IPython.embed()

        print("      Writing {}".format(shape_id))
        write_to_tfrecord(ds, join(shapenet_record_path, shape_id + ".tfrecord"))

def load_shapenet(shapes = "all"):
    if shapes == "all":
        shapes = [f[:-9] for f in os.listdir(shapenet_record_path)
                  if f.endswith(".tfrecord")]

    ds = None
    for shape in shapes:
        fp = join(shapenet_record_path, shape + ".tfrecord")
        if ds:
            ds.concatenate(read_from_record(fp))
        else:
            ds = read_from_record(fp)
    return ds



def write_to_tfrecord(dataset, record_file):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _float_feature(value):
        """Returns a float_list from a float / double."""
        # IPython.embed()
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    with tf.io.TFRecordWriter(record_file) as writer:
        for elem in dataset:
            # IPython.embed()
            # IPython.embed()
            feature={
                'voxelgrid': _bytes_feature(tf.io.serialize_tensor(elem[0]).numpy())
                }
            # IPython.embed()
            features = tf.train.Features(feature=feature)
            example = tf.train.Example(features=features)

            writer.write(example.SerializeToString())
            # writer.write(tf.io.serialize_tensor(elem[0]).numpy())
            

            

def read_from_record(record_file):
    print("hi")
    raw_dataset = tf.data.TFRecordDataset(record_file)

    voxelgrid_description = {
        'voxelgrid': tf.io.FixedLenFeature([], tf.string)
    }


    def _get_shape(_raw_dataset):
        e = next(_raw_dataset.__iter__())
        example = tf.io.parse_single_example(e, voxelgrid_description)
        t = tf.io.parse_tensor(example['voxelgrid'], tf.bool)
        return t.shape
    shape = _get_shape(raw_dataset)
        

    def _parse_voxelgrid_function(example_proto):
        # Parse the input tf.Example proto using the dictionary above.
        example = tf.io.parse_single_example(example_proto, voxelgrid_description)
        t = tf.io.parse_tensor(example['voxelgrid'], tf.bool)
        t.set_shape(shape)
        return (t,t)


    parsed_dataset = raw_dataset.map(_parse_voxelgrid_function)

    # elem = next(parsed_dataset.__iter__())
    # IPython.embed()

    
    return parsed_dataset




if __name__=="__main__":
    print("Not meant to be executed as main script")
    
