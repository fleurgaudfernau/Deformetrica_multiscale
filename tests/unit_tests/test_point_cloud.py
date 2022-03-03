import os
import unittest

import numpy as np


import deformetrica as dfca

# Tests are done both in 2d and 3d.
from . import unit_tests_data_dir


class PointCloudTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        self.points = np.array([[20., 20.], [20., 30.], [60., 20.]])
        self.points3D = np.array([np.concatenate([elt, [0.]]) for elt in self.points])

    def _read_point_cloud(self, path, dimension):
        reader = dfca.io.DeformableObjectReader()
        object = reader.create_object(path, "PointCloud", dimension=dimension)
        return object

    def test_read_point_cloud(self):
        self._test_read_point_cloud_with_dimension(2)
        self._test_read_point_cloud_with_dimension(3)

    def _test_read_point_cloud_with_dimension(self, dim):
        """
        Reads an example vtk file and checks a few points and triangles
        """
        poly_line = self._read_point_cloud(os.path.join(unit_tests_data_dir, "point_cloud.vtk"), dim)
        points = poly_line.get_points()
        if dim == 2:
            self.assertTrue(np.allclose(self.points, points[:3], rtol=1e-05, atol=1e-08))
        elif dim == 3:
            self.assertTrue(np.allclose(self.points3D, points[:3], rtol=1e-05, atol=1e-08))

    def test_set_points_point_cloud(self):
        self._test_read_point_cloud_with_dimension(2)
        self._test_read_point_cloud_with_dimension(3)

    def _test_set_points_point_cloud_with_dimension(self, dim):
        """
        Reads a vtk
        Set new point coordinates using SetPoints
        Asserts the points sent by GetData of the object are the new points
        """
        # Settings().dimension = dim

        poly_line = self._read_point_cloud(os.path.join(unit_tests_data_dir, "skull.vtk"), dim)
        points = poly_line.get_points()
        random_shift = np.random.uniform(0,1,points.shape)
        deformed_points = points + random_shift
        poly_line.set_points(deformed_points)
        deformed_points_2 = poly_line.get_points()
        self.assertTrue(np.allclose(deformed_points, deformed_points_2, rtol=1e-05, atol=1e-08))

# TODO coverage to be added when weights are used for the point cloud.
