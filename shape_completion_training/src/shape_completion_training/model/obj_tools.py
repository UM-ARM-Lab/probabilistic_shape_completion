#!/usr/bin/env python

# import pymesh
# import pywavefront
import pyassimp
import numpy as np
import os


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

    y_rotations = range(0,360, 5)
    # x_trans = [-0.1, 0, 0.1]
    # y_trans = [-0.1, 0, 0.1]
    # z_trans = [-0.1, 0, 0.1]
    x_trans = [0]
    y_trans = [0]
    z_trans = [0]
    

    for yr in y_rotations:
        for x in x_trans:
            for y in y_trans:
                for z in z_trans:
                    transform(scene, yr, x, y, z, savepath)

def transform(scene, y_rot, x_trans, y_trans, z_trans, savepath):
    m = roty(y_rot * 3.1415/180)
    scene.mRootNode.contents.mTransformation.a1 = m[0,0]
    scene.mRootNode.contents.mTransformation.a2 = m[0,1]
    scene.mRootNode.contents.mTransformation.a3 = m[0,2]
    scene.mRootNode.contents.mTransformation.b1 = m[1,0]
    scene.mRootNode.contents.mTransformation.b2 = m[1,1]
    scene.mRootNode.contents.mTransformation.b3 = m[1,2]
    scene.mRootNode.contents.mTransformation.c1 = m[2,0]
    scene.mRootNode.contents.mTransformation.c2 = m[2,1]
    scene.mRootNode.contents.mTransformation.c3 = m[2,2]
    scene.mRootNode.contents.mTransformation.a4 = x_trans
    scene.mRootNode.contents.mTransformation.b4 = y_trans
    scene.mRootNode.contents.mTransformation.c4 = z_trans

    fn = "model_augmented_{:1.1f}_{:1.1f}_{:1.1f}_{:03d}.obj".format(x_trans, y_trans, z_trans,
                                                                     y_rot)
    savename = os.path.join(savepath, fn)
    pyassimp.export(scene, savename, 'obj')
    

def rotx(rad):
    s = np.sin(rad)
    c = np.cos(rad)

    return np.array([[1, 0, 0, 0],
                     [0, c, -s, 0],
                     [0, s, c, 0],
                     [0, 0, 0, 1]])

def roty(rad):
    s = np.sin(rad)
    c = np.cos(rad)
    return np.array([[c, 0, s, 0],
                     [0, 1, 0, 0],
                     [-s, 0, c, 0],
                     [0, 0, 0, 1]])

def rotz(rad):
    s = np.sin(rad)
    c = np.cos(rad)
    return np.array([[c, -s, 0, 0],
                     [s, c, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    




# def load(filepath):
#     scene = pywavefront.Wavefront(filepath, create_materials=True, parse=False, strict=True)
#     IPython.embed()

if __name__=="__main__":
    print("hi")
