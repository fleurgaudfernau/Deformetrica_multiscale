import os
import torch
import platform

import unittest
from tests.functional_tests.functional_test import FunctionalTest


class LongitudinalAtlasHippocampi(FunctionalTest):
    """
    Methods with names starting by "test" will be run.
    """

    def test_configuration_1(self):
        self.run_configuration(os.path.abspath(__file__), 'output__1', 'output_saved__1',
                               'model__1.xml', 'data_set.xml', 'optimization_parameters__1.xml')

    def test_configuration_2(self):
        self.run_configuration(os.path.abspath(__file__), 'output__2', 'output_saved__2',
                               'model__2.xml', 'data_set.xml', 'optimization_parameters__2.xml')

    def test_configuration_3(self):
        self.run_configuration(os.path.abspath(__file__), 'output__3', 'output_saved__3',
                               'model__3.xml', 'data_set.xml', 'optimization_parameters__3.xml')

    # @unittest.skipIf(not torch.cuda.is_available(), 'This test should be run with CUDA.')
    # @unittest.skipIf(platform in ['darwin'], 'This test should only run on linux.')
    # def test_configuration_4(self):
    #     self.run_configuration(os.path.abspath(__file__), 'output__4', 'output_saved__4',
    #                            'model__4.xml', 'data_set.xml', 'optimization_parameters__4.xml',
    #                            command='initialize', precision=1e-4)

    # def test_configuration_5(self):
    #     self.run_configuration(os.path.abspath(__file__), 'output__5', 'output_saved__5',
    #                            'model__5.xml', 'data_set.xml', 'optimization_parameters__5.xml')
