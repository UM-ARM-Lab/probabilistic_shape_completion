#! /usr/bin/env python

import sys
import os
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import obj_tools
from model import data_tools
import IPython
import subprocess
from itertools import izip_longest
from threading import Thread

"""
NOTE:
If running over ssh, need to start a virtual screen
https://www.patrickmin.com/binvox/

Xvfb :99 -screen 0 640x480x24 &
export DISPLAY=:99


Then run binvox with the -pb option 

"""
def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)

def augment_category(object_path, num_threads = 1):

    # shapes = ['a1d293f5cc20d01ad7f470ee20dce9e0']
    # shapes = ['214dbcace712e49de195a69ef7c885a4']
    shape_ids = os.listdir(object_path)
    shape_ids.sort()


    # i = 0
    # for group in grouper(num_threads, shape_ids):
    #     # IPython.embed()
    #     threads = []
    #     for shape in group:
    #         i+= 1
    #         fp = join(object_path, shape, "models")
    #         print("{:03d}/{} Augmenting {}".format(i, len(shape_ids), shape))
    #         thread = Thread(target = augment_shape, args = (fp, ))
    #         thread.start()
    #         threads.append(thread)
    #     for thread in threads:
    #         thread.join()
    
    
    i=0;
    for shape in shape_ids:
        i+=1

        print("{:03d}/{} Augmenting {}".format(i, len(shape_ids), shape))
        
        fp = join(object_path, shape, "models")
        augment_shape(fp, num_threads)


"""
Augments the model at the filepath

filepath should end with the "models" folder
Augmentation involves rotatin the model and converting all rotations to .binvox files
"""
def augment_shape(filepath, num_threads):
    fp = filepath

    if fp is None:
        return
    
    old_files = [f for f in os.listdir(fp) if f.startswith("model_augmented")]
    for f in old_files:
        os.remove(join(fp, f))



    obj_path = join(fp, "model_normalized.obj")
    obj_tools.augment(obj_path)

    augmented_obj_files = [f for f in os.listdir(fp)
                           if f.startswith('model_augmented')
                           if f.endswith('.obj')]

    threaded_binvox_object_files(fp, augmented_obj_files, num_threads)
    # for f in augmented_obj_files:
    #     binvox_object_file(join(fp, f))

    #Cleanup large model files
    old_files = [f for f in os.listdir(fp)
                 if f.startswith("model_augmented")
                 if not f.endswith(".binvox")]
    for f in old_files:
        os.remove(join(fp, f))

def threaded_binvox_object_files(fp, augmented_obj_files, num_threads):
    for group in grouper(num_threads, augmented_obj_files):
        threads = []
        for aug_obj in group:
            thread = Thread(target=binvox_object_file, args=(join(fp, aug_obj), ))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()

        

"""
Runs binvox on the input obj file
"""
def binvox_object_file(fp):

    #TODO Hardcoded binvox path
    binvox_str = "~/useful_scripts/binvox -dc -pb -down -down -dmin 2 {}".format(fp)

    #Fast but inaccurate
    wire_binvox_str = "~/useful_scripts/binvox -e -pb -down -down -dmin 1 {}".format(fp)
    cuda_binvox_str = "~/useful_scripts/cuda_voxelizer -s 64 -f {}".format(fp)

    fp_base = fp[:-4]
    
    with open(os.devnull, 'w') as FNULL:
        subprocess.call(binvox_str, shell=True, stdout=FNULL)
        os.rename(fp_base + ".binvox", fp_base + ".mesh.binvox")
        
        subprocess.call(wire_binvox_str, shell=True, stdout=FNULL)
        os.rename(fp_base + ".binvox", fp_base + ".wire.binvox")
        
        # subprocess.call(cuda_binvox_str, shell=True, stdout=FNULL)

        


if __name__=="__main__":
    gt_obj_fp = "/home/bsaund/catkin_ws/src/mps_shape_completion/shape_completion_training/data/ShapeNetCore.v2_augmented/03797390/a1d293f5cc20d01ad7f470ee20dce9e0/models/model_normalized.obj"


    sn_path = join(data_tools.cur_path, "../data/ShapeNetCore.v2_augmented")
    sn_path = join(sn_path, data_tools.shape_map['mug'])


    augment_category(sn_path, 18)




