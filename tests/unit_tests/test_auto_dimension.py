import os
import unittest

import deformetrica as dfca
from . import unit_tests_data_dir


class AutomaticDimensionDetectionTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        self.object_reader = dfca.io.DeformableObjectReader()

    def tearDown(self):
        super().tearDown()

    def test_auto_dimension_2D_vtk(self):
        _, dimension, _ = self.object_reader.read_file(os.path.join(unit_tests_data_dir, 'bonhomme.vtk'), extract_connectivity=True)
        self.assertEqual(2, dimension)
        _, dimension = self.object_reader.read_file(os.path.join(unit_tests_data_dir, 'point_cloud.vtk'), extract_connectivity=False)
        self.assertEqual(2, dimension)
        _, dimension, _ = self.object_reader.read_file(os.path.join(unit_tests_data_dir, 'skull.vtk'), extract_connectivity=True)
        self.assertEqual(2, dimension)

    def test_auto_dimension_3D_vtk(self):
        _, dimension, _ = self.object_reader.read_file(os.path.join(unit_tests_data_dir, 'hippocampus.vtk'), extract_connectivity=True)
        self.assertEqual(3, dimension)
        _, dimension, _ = self.object_reader.read_file(os.path.join(unit_tests_data_dir, 'hippocampus_2.vtk'), extract_connectivity=True)
        self.assertEqual(3, dimension)

    def test_auto_dimension_create_object(self):
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'bonhomme.vtk'), 'landmark')
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'point_cloud.vtk'), 'landmark')
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'skull.vtk'), 'polyline')
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'digit_2_sample_1.png'), 'image')
        self.assertEqual(2, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'hippocampus.vtk'), 'SurfaceMesh')
        self.assertEqual(3, o.dimension)
        o = self.object_reader.create_object(os.path.join(
            unit_tests_data_dir, 'polyline_different_format.vtk'), 'polyline')
        self.assertEqual(3, o.dimension)
