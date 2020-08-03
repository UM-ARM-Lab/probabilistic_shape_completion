# Predicting Diverse Plausible Shape Completions from Ambiguous Depth Images

This package provides a neural network that takes in a grid of visible occupied voxels from a single view and outputs a grid of the estimated 3D voxels, thus "completing the shape". Running multiple inference passes with the same input will generate different, yet plausible completions. Depending on the ambiguity of the input the completions may all be quite similar, or vary noticably.


### Prerequisites
The code is developed and tested on
- [`CUDA`](https://developer.nvidia.com/cuda-toolkit) 10.2 
- [`cuDNN`](https://developer.nvidia.com/rdp/cudnn-archive) 7.6
- [`Python`](https://www.python.org) 2.7.12
- [`TensorFlow`](https://github.com/tensorflow/tensorflow) 2.1
- [`numpy`](http://www.numpy.org/) 1.14.2
- `ROS` [kinetic](http://wiki.ros.org/kinetic) or [melodic](http://wiki.ros.org/melodic)
- For visualization in RViz: https://github.com/bsaund/rviz_text_selection_panel and https://github.com/bsaund/rviz_voxelgrid_visuals



## Structure
 - `shape_completion_training`: Generating, training, evaluating the shape completion model and baselines
 - `shape_completion_visualization`: Scripts for viewing shape datasets and completions in RViz
