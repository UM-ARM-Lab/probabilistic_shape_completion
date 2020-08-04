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
 
 Packages are structed as ROS packages. Scripts to be run (e.g. in a terminal) are located in the `scripts` folders.
 
 Note, the PSSNet in the paper is named "NormalizingAE" is this code.
 
 
 ## How to fully recreate paper results (Note, will take several days):
 1. Install all dependencies listed above
 2. Download `shapenet` (account needed) and YCB (`download_ycb.py`) and place in `./shape_completion_training/data
 3. Augment the datasets by rotating and computing voxelgrids using the scripts `augment_shapenet.py` and `augment_ycb.py`
 4. Preprocess datasets into tensorflow `TFDatasets`: `preprocess_shapenet.py` and `preprocess_ycb.py`
 5. Train networks using `./train.py --group [Trial Name]`, where `Trial Name`s can be found in `defaults.py` and specify the network, dataset, and other trial parameters. (e.g. `./train.py --group NormalizingAE`)
 6. Compute plausibe sets `./shapenet_plausibilities.py`, `ycb_plausibilities.py`. Note, both of these uses sharded datasets so they can be computed in parallel. If compting in parallel, combine by using the `--combine_shards` option
 7. Evaluate using `./evaluate`. You can edit the model names if not evaluating all models.
 
 ## How to view results in RViz
 1. Open rviz with the `./shape_completion_visualization/shape_completion.rviz` config file (Note: Will need to build `catkin` packages first to have the necessary rviz plugins)
 2. run `./shape_completion_visualization/data_publisher.py` to view the data alone. Pass `--trial [Trial Path]` option to view completions using a network 
