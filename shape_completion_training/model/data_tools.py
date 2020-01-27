#! /usr/bin/env python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
# from shape_completion_training import binvox_rw
import binvox_rw
import numpy as np
import IPython


obj = "025_mug"
base_path = "/home/bsaund/tmp/shape_completion/instance_0619_new_model/data_occ/"
gt_path = base_path + obj + "/gt/"
occ_path = base_path + obj + "/train_x_occ/"
non_occ_path = base_path + obj + "/non_occupy/"

record_name = "./data/" + obj + ".tfrecord"
    

"Loads data from file. From a TF_Record, if asked"
def load_data(from_record=False):


    if from_record:
        return tf.data.TFRecordReader(record_name)
    

    files = [f for f in os.listdir(occ_path)]
    files.sort()

    data = []

    for filename in files:
        prefix = filename.split("occupy")[0]
        print(prefix)

        with open(os.path.join(gt_path,prefix + "gt.binvox")) as f:
            gt_vox = binvox_rw.read_as_3d_array(f).data

        data.append(gt_vox)

    data = np.array(data)
    data = np.expand_dims(data, axis=4)
    # IPython.embed()
    ds = tf.data.Dataset.from_tensor_slices((data, data))
    # write_to_file(ds)
    return ds



def write_to_file(dataset):

    
    writer = tf.data.experimental.TFRecordWriter(record_name)

    
    IPython.embed()
    writer.write(ds)



if __name__=="__main__":
    print("Not meant to be executed as main script")
    
