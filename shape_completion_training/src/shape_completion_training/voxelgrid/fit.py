from shape_completion_training.voxelgrid import conversions
import pcl
import numpy as np


def icp(voxelgrid_to_transform, template_voxelgrid, scale, max_iter=10):
    """
    Returns a voxelgrid fitted to the template using ICP
    @param scale:
    @param voxelgrid_to_transform:
    @param template_voxelgrid:
    @return:
    """
    pt_0 = conversions.voxelgrid_to_pointcloud(template_voxelgrid, scale=scale)
    pt_1 = conversions.voxelgrid_to_pointcloud(voxelgrid_to_transform, scale=scale)

    source = pcl.PointCloud(pt_0.astype(np.float32))
    target = pcl.PointCloud(pt_1.astype(np.float32))
    icp = source.make_IterativeClosestPoint()
    converged, transf, estimate, fitness = icp.icp(
        source, target, max_iter=max_iter)

    T = np.linalg.inv(transf)
    vg_icp = conversions.transform_voxelgrid(voxelgrid_to_transform, T, scale=scale)
    return vg_icp
