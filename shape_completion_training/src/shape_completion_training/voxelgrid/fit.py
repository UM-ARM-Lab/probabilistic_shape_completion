from __future__ import print_function
from shape_completion_training.voxelgrid import conversions
import pcl
import numpy as np


def icp(voxelgrid_to_transform, template_voxelgrid, scale, max_iter=10, downsample = 1):
    """
    Returns a voxelgrid fitted to the template using ICP
    @param max_iter: maximum iterations of ICP
    @param downsample: downsampling factor so fewer points are fit during ICP
    @param scale:
    @param voxelgrid_to_transform:
    @param template_voxelgrid:
    @return:
    """

    vg_to_transform_downsampled = conversions.downsample(voxelgrid_to_transform, downsample)
    template_voxelgrid = conversions.downsample(template_voxelgrid, downsample)

    pt_0 = conversions.voxelgrid_to_pointcloud(template_voxelgrid, scale=scale)
    pt_1 = conversions.voxelgrid_to_pointcloud(vg_to_transform_downsampled, scale=scale)

    source = pcl.PointCloud(pt_0.astype(np.float32))
    target = pcl.PointCloud(pt_1.astype(np.float32))
    icp = source.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(
        source, target, max_iter=max_iter)

    T = np.linalg.inv(transf)
    vg_icp = conversions.transform_voxelgrid(voxelgrid_to_transform, T, scale=scale)
    return vg_icp
