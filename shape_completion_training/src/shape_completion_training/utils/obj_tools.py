#!/usr/bin/env python

# import pymesh
# import pywavefront
import pyassimp
import numpy as np
import os
from shape_completion_training.utils.matrix_math import rotzyx

"""
I expect somewhere there is an existing python library that will load my obj files nicely. I have not found such a library.
I tried PyMesh, which seems to work fine but wont install
I tried pywavefront which error when loaded these obj files
sudo apt install python-pyassimp

"""

"""
Possible useful commands:

binvox command: binvox -aw -dc -pb -down 2 model_normalized.obj 

docker run -it --rm -v `pwd`:/models pymesh/pymesh /models/augment_shapenet.py


"""


def augment(filename):
    scene = pyassimp.load(filename)

    savepath = os.path.dirname(filename)

    x_rotations = [-90, 0, 90]
    # y_rotations = range(0, 360, 5)
    # z_rotations = [-150, -120, -90, -60, -45, -30, 0, 30, 45, 60, 90, 120, 150, 180]
    y_rotations = range(0,360, 60)
    z_rotations = [-90, 0, 90]

    for xr in x_rotations:
        for yr in y_rotations:
            for zr in z_rotations:
                transform(scene, xr, yr, zr, savepath)


def transform(scene, x_rot, y_rot, z_rot, savepath):
    m = rotzyx(x_rot, y_rot, z_rot, degrees=True)
    scene.mRootNode.contents.mTransformation.a1 = m[0, 0]
    scene.mRootNode.contents.mTransformation.a2 = m[0, 1]
    scene.mRootNode.contents.mTransformation.a3 = m[0, 2]
    scene.mRootNode.contents.mTransformation.b1 = m[1, 0]
    scene.mRootNode.contents.mTransformation.b2 = m[1, 1]
    scene.mRootNode.contents.mTransformation.b3 = m[1, 2]
    scene.mRootNode.contents.mTransformation.c1 = m[2, 0]
    scene.mRootNode.contents.mTransformation.c2 = m[2, 1]
    scene.mRootNode.contents.mTransformation.c3 = m[2, 2]
    scene.mRootNode.contents.mTransformation.a4 = 0
    scene.mRootNode.contents.mTransformation.b4 = 0
    scene.mRootNode.contents.mTransformation.c4 = 0

    fn = "model_augmented_{:03d}_{:03d}_{:03d}.obj".format(x_rot, y_rot, z_rot)
    savename = os.path.join(savepath, fn)
    pyassimp.export(scene, savename, 'obj')


# def load(filepath):
#     scene = pywavefront.Wavefront(filepath, create_materials=True, parse=False, strict=True)
#     IPython.embed()

if __name__ == "__main__":
    print("hi")
