import numpy as np
import pcl
from numpy import cos, sin
from shape_completion_training.tests.setup_data_for_unit_tests import load_test_files
from mps_shape_completion_visualization.quick_publish import publish_voxelgrid, publish_object_transform
from shape_completion_training.voxelgrid import conversions
from shape_completion_training.voxelgrid import fit
import rospy

from scipy.spatial.transform import Rotation
from pcl import IterativeClosestPoint
import unittest

p = pcl.PointCloud(10)  # "empty" point cloud
a = np.asarray(p)  # NumPy view on the cloud
a[:] = 0  # fill with zeros
print(p[3])  # prints (0.0, 0.0, 0.0)
a[:, 0] = 1  # set x coordinates to 1
print(p[3])  # prints (1.0, 0.0, 0.0)


class ICP():
    def setUp(self):
        # Check if ICP can find a mild rotation.
        theta = [0, 0, .5]
        trans = [.1, -.1, 2]
        rot_x = [[1, 0, 0],
                 [0, cos(theta[0]), -sin(theta[0])],
                 [0, sin(theta[0]), cos(theta[0])]]
        rot_y = [[cos(theta[1]), 0, sin(theta[1])],
                 [0, 1, 0],
                 [-sin(theta[1]), 0, cos(theta[1])]]
        rot_z = [[cos(theta[2]), -sin(theta[1]), 0],
                 [sin(theta[2]), cos(theta[1]), 0],
                 [0, 0, 1]]
        transform = np.dot(rot_x, np.dot(rot_y, rot_z))

        # transform = Rotation.from_euler('zyx', [theta])

        source = np.random.RandomState(42).randn(900, 3)
        self.source = pcl.PointCloud(source.astype(np.float32))

        target = np.dot(source, transform)
        target += np.array(trans)
        self.target = pcl.PointCloud(target.astype(np.float32))

    def testICP(self):
        icp = self.source.make_IterativeClosestPoint()
        converged, transf, estimate, fitness = icp.icp(
            self.source, self.target, max_iter=1000)

        print("Converged: {}".format(converged))
        print(transf)
        # self.assertTrue(converged is True)
        # # fail: pcl 1.9.1
        # # self.assertLess(fitness, .1)
        #
        # self.assertTrue(isinstance(transf, np.ndarray))
        # self.assertEqual(transf.shape, (4, 4))

        # self.assertFalse(np.any(transf[:3] == 0))


def test_icp():
    icp = ICP()
    icp.setUp()
    icp.testICP()


if __name__ == "__main__":
    scale = 0.1
    rospy.init_node("wip_pcl")
    d = load_test_files()
    publish_voxelgrid(d[0], "gt_voxel_grid")


    # pt_0 = conversions.voxelgrid_to_pointcloud(d[0], scale=scale)
    # pt_1 = conversions.voxelgrid_to_pointcloud(vg_other, scale=scale)
    #
    #
    # publish_voxelgrid(vg_icp, "sampled_occ_voxel_grid")
    for i, vg in enumerate(d):
        # vg_other = d[2]
        print("fitting {}".format(i))
        publish_voxelgrid(fit.icp(vg, d[0], scale, max_iter=10), "sampled_occ_voxel_grid")
        publish_voxelgrid(vg, "predicted_occ_voxel_grid")
        rospy.sleep(1)
