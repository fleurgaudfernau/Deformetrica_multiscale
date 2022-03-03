#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
import unittest

from tests.unit_tests.test_api import API
from tests.unit_tests.test_array_readers_and_writers import ArrayReadersAndWritersTests
from tests.unit_tests.test_attachments import DistanceTests
from tests.unit_tests.test_auto_dimension import AutomaticDimensionDetectionTests
from tests.unit_tests.test_kernel_factory import KeopsVersusCuda, KernelFactoryTest, TorchKernelTest, KeopsKernelTest
from tests.unit_tests.test_parallel_transport import ParallelTransportTests
from tests.unit_tests.test_point_cloud import PointCloudTests
from tests.unit_tests.test_poly_line import PolyLineTests
from tests.unit_tests.test_shooting import ShootingTests
from tests.unit_tests.test_surface_mesh import SurfaceMeshTests

TEST_MODULES = [API, KernelFactoryTest, TorchKernelTest, KeopsKernelTest, KeopsVersusCuda,
                ParallelTransportTests, DistanceTests, ArrayReadersAndWritersTests,
                PolyLineTests, PointCloudTests, SurfaceMeshTests, ShootingTests,
                AutomaticDimensionDetectionTests]

# TEST_MODULES = [ParallelTransportTests]


def main():
    # import logging
    # logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler())
    # logger.setLevel(logging.DEBUG)

    success = True

    for t in TEST_MODULES:
        res = unittest.TextTestRunner(verbosity=2).run(unittest.TestLoader().loadTestsFromTestCase(t))
        success = success and res.wasSuccessful()

    # logger.info(success)
    if not success:
        sys.exit('Test failure !')


if __name__ == '__main__':
    main()
