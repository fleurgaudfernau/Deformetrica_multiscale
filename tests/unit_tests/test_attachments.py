import os
import unittest

import torch
import deformetrica as dfca
from torch.autograd import Variable

# Tests a few distances computations (current and varifold) and compares them to C++ version
from . import unit_tests_data_dir


class DistanceTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        self.tensor_scalar_type = torch.DoubleTensor
        self.tensor_integer_type = torch.LongTensor
        self.kernel = dfca.kernels.factory('torch', kernel_width=10.)
        # self.kernel = kernel_factory.factory('keops', kernel_width=10.)  # Duplicate the tests for both kernels ?
        self.multi_attach = dfca.attachments.MultiObjectAttachment('', self.kernel)

    def _read_surface_mesh(self, path):
        reader = dfca.io.DeformableObjectReader()
        object = reader.create_object(path, "SurfaceMesh", dimension=3)
        return object

    def _read_poly_line(self, path):
        reader = dfca.io.DeformableObjectReader()
        object = reader.create_object(path, "PolyLine", dimension=2)
        return object

    def test_surface_mesh_varifold_distance_to_self_is_zero(self):
        surf = self._read_surface_mesh(os.path.join(unit_tests_data_dir, "hippocampus.vtk"))
        points = torch.from_numpy(surf.get_points()).type(self.tensor_scalar_type)
        varifold_distance = self.multi_attach.varifold_distance(points, surf, surf, self.kernel).data.numpy()
        self.assertTrue(abs(varifold_distance) < 1e-7)

    def test_surface_mesh_current_distance_to_self_is_zero(self):
        surf = self._read_surface_mesh(os.path.join(unit_tests_data_dir, "hippocampus.vtk"))
        points = Variable(torch.from_numpy(surf.get_points()).type(self.tensor_scalar_type))
        current_distance = self.multi_attach.current_distance(points, surf, surf, self.kernel).data.numpy()
        self.assertTrue(abs(current_distance) < 1e-7)

    def test_varifold_distance_on_surface_mesh_is_equal_to_old_deformetrica(self):
        source = self._read_surface_mesh(os.path.join(unit_tests_data_dir, "hippocampus.vtk"))
        target = self._read_surface_mesh(os.path.join(unit_tests_data_dir, "hippocampus_2.vtk"))
        points_source = Variable(torch.from_numpy(source.get_points()).type(self.tensor_scalar_type))
        varifold_distance = self.multi_attach.varifold_distance(points_source, source, target, self.kernel).data.numpy()
        old_deformetrica_varifold_distance = 10662.59732 # C++ version
        self.assertTrue(abs(varifold_distance - old_deformetrica_varifold_distance)/abs(old_deformetrica_varifold_distance) < 1e-5)

    def test_current_distance_on_surface_mesh_is_equal_to_old_deformetrica(self):
        source = self._read_surface_mesh(os.path.join(unit_tests_data_dir, "hippocampus.vtk"))
        target = self._read_surface_mesh(os.path.join(unit_tests_data_dir, "hippocampus_2.vtk"))
        points_source = Variable(torch.from_numpy(source.get_points()).type(self.tensor_scalar_type))
        current_distance = self.multi_attach.current_distance(points_source, source, target, self.kernel).data.numpy()
        old_deformetrica_current_distance = 3657.504384
        self.assertTrue(abs(current_distance - old_deformetrica_current_distance)/abs(old_deformetrica_current_distance) < 1e-5)

    def _test_poly_line_current_distance_to_self_is_zero(self):
        poly = self._read_poly_line(os.path.join(unit_tests_data_dir, "skull.vtk"))
        points = Variable(torch.from_numpy(poly.get_points()).type(self.tensor_scalar_type))
        current_distance = self.multi_attach.current_distance(points, poly, poly, self.kernel).data.numpy()
        self.assertTrue(abs(current_distance) < 1e-10)

    def test_poly_line_current_distance_to_self_is_zero(self):
        self._test_poly_line_current_distance_to_self_is_zero()
        self._test_poly_line_current_distance_to_self_is_zero()
