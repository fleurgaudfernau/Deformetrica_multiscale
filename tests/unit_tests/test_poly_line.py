import os
import unittest

import numpy as np

import deformetrica as dfca

import shutil

# Tests are done both in 2 and 3d.
from . import unit_tests_data_dir


class PolyLineTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """

    def setUp(self):
        self.points = np.array([[16.463592, -34.480583], [16.463592, -28.980583], [15.463592, -25.980583]])
        self.points3D = np.array([np.concatenate([elt, [0.]]) for elt in self.points])
        self.first_line = np.array([0, 1])

    def _read_poly_line(self, path, dimension):
        reader = dfca.io.DeformableObjectReader()
        object = reader.create_object(path, "PolyLine", dimension=dimension)
        return object

    def test_read_poly_line(self):
        self._test_read_poly_line_with_dimension(2)
        self._test_read_poly_line_with_dimension(3)
        # this one has a slightly different format of vtk. This test covers the read/write of this different format.
        self._test_read_poly_line_different_format(2)
        self._test_read_poly_line_different_format(3)

    def _test_read_poly_line_with_dimension(self, dim):
        """
        Reads an example vtk file and checks a few points and triangles
        """
        poly_line = self._read_poly_line(os.path.join(unit_tests_data_dir, "skull.vtk"), dim)
        points = poly_line.get_points()
        if dim == 2:
            self.assertTrue(np.allclose(self.points, points[:3], rtol=1e-05, atol=1e-08))
        elif dim == 3:
            self.assertTrue(np.allclose(self.points3D, points[:3], rtol=1e-05, atol=1e-08))
        other_first_triangle = poly_line.connectivity[0]
        self.assertTrue(np.allclose(self.first_line, other_first_triangle))

    def _test_read_poly_line_different_format(self, dim):
        poly_line = self._read_poly_line(os.path.join(unit_tests_data_dir, "polyline_different_format.vtk"), dim)
        points = poly_line.get_points()
        lines = poly_line.connectivity
        tmp_folder = os.path.join(os.path.dirname(__file__), 'tmp')
        os.mkdir(tmp_folder)
        poly_line.write(os.path.join(tmp_folder, 'output'),
                        os.path.join(tmp_folder, 'written_polyline_different_format.vtk'), points)
        re_read_poly_line = self._read_poly_line(os.path.join(tmp_folder, 'written_polyline_different_format.vtk'), dim)
        shutil.rmtree(tmp_folder)
        re_read_points = re_read_poly_line.get_points()
        re_read_lines = re_read_poly_line.connectivity
        self.assertTrue(np.allclose(points, re_read_points))
        self.assertTrue(np.allclose(lines, re_read_lines))

    def test_set_points_poly_line(self):
        self._test_read_poly_line_with_dimension(2)
        self._test_read_poly_line_with_dimension(3)

    def _test_set_points_poly_line_with_dimension(self, dim):
        """
        Reads a vtk
        Set new point coordinates using SetPoints
        Asserts the points sent by GetData of the object are the new points
        """
        poly_line = self._read_poly_line(os.path.join(unit_tests_data_dir, "skull.vtk"), dim)
        points = poly_line.get_points()
        random_shift = np.random.uniform(0, 1, points.shape)
        deformed_points = points + random_shift
        poly_line.set_points(deformed_points)
        deformed_points_2 = poly_line.get_points()
        self.assertTrue(np.allclose(deformed_points, deformed_points_2, rtol=1e-05, atol=1e-08))

    def _test_centers_and_normals_with_dimension(self, dim):
        """
        Tests the computation of centers and normals on the hippocampus, on all triangles
        """
        poly_line = self._read_poly_line(os.path.join(unit_tests_data_dir, "skull.vtk"), dim)
        pts = poly_line.get_points()
        lines = poly_line.connectivity
        centers, normals = poly_line.get_centers_and_normals()
        self.assertTrue(centers.shape == (len(lines), dim))
        for i, line in enumerate(lines):
            pts_line = [pts[j] for j in line]
            center = np.mean(pts_line, 0)
            normal = pts_line[1] - pts_line[0]
            self.assertTrue(np.allclose(center, centers.data.numpy()[i]))
            self.assertTrue(np.allclose(normal, normals.data.numpy()[i]))

    def test_centers_and_normals(self):
        self._test_centers_and_normals_with_dimension(2)
        self._test_centers_and_normals_with_dimension(3)
