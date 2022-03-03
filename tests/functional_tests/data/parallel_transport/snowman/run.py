import os
import torch
import unittest

from tests.functional_tests.functional_test import FunctionalTest


class ParallelTransportSnowman(FunctionalTest):
    """
    Methods with names starting by "test" will be run.
    """

    def test_configuration_1(self):
        self.run_configuration(os.path.abspath(__file__), 'output__1', 'output_saved__1', 'model__1.xml', None, 'optimization_parameters__1.xml', command='compute')

    def test_configuration_2(self):
        self.run_configuration(os.path.abspath(__file__), 'output__2', 'output_saved__2', 'model__2.xml', None, 'optimization_parameters__2.xml', command='compute')

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_configuration_3(self):
        self.run_configuration(os.path.abspath(__file__), 'output__3', 'output_saved__3', 'model__3.xml', None, 'optimization_parameters__3.xml', command='compute')
