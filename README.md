# Shape-completion

MPS Shape Completion is a neural network that takes in a grid of visible occupied voxels and outputs a grid of voxels the NN believes are occupied, thus "completing the shape". 
This has been used in the "Moving Piles of Stuff" project.

[![Build Status](https://travis-ci.com/UM-ARM-Lab/mps_shape_completion.svg?branch=master)](https://travis-ci.com/UM-ARM-Lab/mps_shape_completion)

## Setup

There are multiple directories.
 - `shape_completion_training`: Generating, training, evaluating the shape completion model
 - `mps_shape_completion`: ROS interface for running shape completion inference
 - `mps_shape_completion_msgs`: ROS message types
 - `mps_shape_completion_visualization`: ROS/RViz tools for viewing voxelgrids

### Prerequisites
The code is developed and tested on
- [`CUDA`](https://developer.nvidia.com/cuda-toolkit) 10.0 
- [`cuDNN`](https://developer.nvidia.com/rdp/cudnn-archive) 7.??
- [`Python`](https://www.python.org) 2.7.12
- [`TensorFlow`](https://github.com/tensorflow/tensorflow) 2.0
- [`numpy`](http://www.numpy.org/) 1.14.2
- [`ROS`](http://wiki.ros.org/kinetic) kinetic
- For visualization in RViz: https://github.com/bsaund/rviz_text_selection_panel

### Initial Demo

1. Clone this repo into a catkin workspace. Rebuild and resource.
2. Download the pretrained model into the `train_mod` folder from [this link](https://drive.google.com/file/d/1Kmij09eHVE3ab5s7Vnp-fI-qOCLei4u0/view?usp=sharing), or use the download script `download_model.sh`
2. View the input files: `./viewvox demo/occupy.binvox`
3. Start a roscore: `roscore`
4. In another terminal start the shape_completion_node `rosrun mps_shape_completion shape_completion_node.py`.
The model should load. This may take a minute.
5. In another terminal window run the ros demo: `rosrun mps_shape_completion ros_demo.py`.
The output should be saved to `demo/output.binvox`
6. View the output: `./viewvox demo/output.binvox`

### Normal use
In normal operation just start a `shape_completion_node.py`. Requests are made either through a ros service call or a ros message via a publisher.

## Overview
- `shape_completion.py`: provides a class that uses the trained model to do shape completion.
- `train_mod/`: contains the model.
- `demo/`: contains sample input in the shape_completion demo. 
- `binvox_rw/`: Python module to read and write .binvox files. [dimatura/binvox-rw-py](https://github.com/dimatura/binvox-rw-py)
- `viewvox`: Reads a 3D voxel file as produced by binvox or thinvox and shows it in a window. [3D voxel model viewer](http://www.patrickmin.com/viewvox/)

- The current model is trained on a subset of objects in ycb dataset and shapenet:

![](https://github.com/UM-ARM-Lab/Shape-completion/blob/master/train_mod/training_set.png)


# Installation
1. Download shapenet and put in the `shape_completion_training/data` folder. Unzip.
2. Install pyassimp `pip install pyassimp=4.1.3`
     https://github.com/assimp/assimp/issues/2343
3. Install binvox
4. If running over ssh, set external display for binvox.
``` 
Xvfb :99 -screen 0 640x480x24 &
export DISPLAY=:99
```

3. Run `augment_shapenet.py`.

