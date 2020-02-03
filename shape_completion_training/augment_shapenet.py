#! /usr/bin/env python

import sys
import os
from os.path import dirname, abspath, join

sc_path = join(dirname(abspath(__file__)), "..")
sys.path.append(sc_path)

from model import obj_tools
from model import data_tools
import docker
import IPython
import subprocess




def augment_object(object_path):

    shapes = ['a1d293f5cc20d01ad7f470ee20dce9e0']

    i=0;

    shapes = os.listdir(object_path)
    for shape in shapes:
    # for shape in shapes:
        i+=1

        print("{:03d}/{} Augmenting {}".format(i, len(shapes), shape))
        
        fp = join(object_path, shape, "models")

        old_files = [f for f in os.listdir(fp) if f.startswith("model_augmented")]
        for f in old_files:
            os.remove(join(fp, f))



        obj_path = join(fp, "model_normalized.obj")
        obj_tools.augment(obj_path)

        augmented_obj_files = [f for f in os.listdir(fp)
                               if f.startswith('model_augmented')
                               if f.endswith('.obj')]
        for f in augmented_obj_files:
            binvox_object_file(join(fp, f))

        #Cleanup large model files
        old_files = [f for f in os.listdir(fp)
                     if f.startswith("model_augmented")
                     if not f.endswith(".binvox")]
        for f in old_files:
            os.remove(join(fp, f))

        

"""
Runs binvox on the input obj file
"""
def binvox_object_file(fp):

    #TODO Hardcoded binvox path
    # binvox_str = "~/useful_scripts/binvox -aw -dc -pb -down -down -dmin 2 {}".format(fp)

    #Fast but inaccurate
    # binvox_str = "~/useful_scripts/binvox -e -pb -down -down -dmin 2 {}".format(fp)
    binvox_str = "~/useful_scripts/cuda_voxelizer -s 64 -f {}".format(fp)

    
    # IPython.embed()

    with open(os.devnull, 'w') as FNULL:
        subprocess.call(binvox_str, shell=True, stdout=FNULL)

        


if __name__=="__main__":
    gt_obj_fp = "/home/bsaund/catkin_ws/src/mps_shape_completion/shape_completion_training/data/ShapeNetCore.v2_augmented/03797390/a1d293f5cc20d01ad7f470ee20dce9e0/models/model_normalized.obj"


    sn_path = join(data_tools.cur_path, "../data/ShapeNetCore.v2_augmented")
    sn_path = join(sn_path, data_tools.shape_map['mug'])


    augment_object(sn_path)




