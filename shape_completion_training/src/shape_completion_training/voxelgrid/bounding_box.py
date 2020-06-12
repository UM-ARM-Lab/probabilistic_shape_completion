"""
Gets a bounding box for shapes
"""
from shape_completion_training.voxelgrid import conversions
import numpy as np


def get_aabb(voxelgrid, scale=0.01):
    """
    Returns the axis aligned bounding box for the voxelgrid
    @param voxelgrid:
    @param scale:
    @return:
    """
    pts = conversions.voxelgrid_to_pointcloud(voxelgrid, scale=scale)
    ub, lb = np.max(pts, axis=0), np.min(pts, axis=0)
    borders = [[lb[0], lb[1], lb[2]],
               [lb[0], lb[1], ub[2]],
               [lb[0], ub[1], lb[2]],
               [lb[0], ub[1], ub[2]],
               [ub[0], lb[1], lb[2]],
               [ub[0], lb[1], ub[2]],
               [ub[0], ub[1], lb[2]],
               [ub[0], ub[1], ub[2]],
               ]
    return np.array(borders)

