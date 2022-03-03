import os
import tempfile
import unittest

from . import unit_tests_data_dir
import deformetrica as dfca
import numpy as np


class ArrayReadersAndWritersTests(unittest.TestCase):
    """
    Methods with names starting by "test" will be run
    """
    def setUp(self):
        self.test_output_file_path = os.path.join(tempfile.gettempdir(), "test_write_3D_array.txt")
        pass

    def tearDown(self):
        # remove created file
        if os.path.isfile(self.test_output_file_path):
            os.remove(self.test_output_file_path)

        super().tearDown()

    def test_read_3D_array(self):
        momenta = dfca.io.read_3D_array(os.path.join(unit_tests_data_dir, "Momenta.txt"))
        self.assertEqual(momenta.shape, (4, 72, 3))
        self.assertTrue(np.allclose(momenta[0, 0], np.array([-0.0313538, -0.00373486, -0.0256917])))
        self.assertTrue(np.allclose(momenta[0, -1], np.array([-0.518624, 1.47211, 0.880905])))
        self.assertTrue(np.allclose(momenta[-1, -1], np.array([2.81286, -0.353167, -2.16408])))

    def test_write_3D_array(self):
        momenta = dfca.io.read_3D_array(os.path.join(unit_tests_data_dir, "Momenta.txt"))
        dfca.io.write_3D_array(momenta, self.test_output_file_path, self.test_output_file_path)
        read = dfca.io.read_3D_array(self.test_output_file_path)
        self.assertTrue(np.allclose(momenta, read))
