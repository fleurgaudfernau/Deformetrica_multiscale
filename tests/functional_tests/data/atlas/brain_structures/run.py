import os
import torch
import unittest

from tests.functional_tests.functional_test import FunctionalTest


class AtlasBrainStructures(FunctionalTest):
    """
    Methods with names starting by "test" will be run.
    """

    def test_configuration_1(self):
        self.run_configuration(os.path.abspath(__file__), 'output__1', 'output_saved__1',
                               'model__1.xml', 'data_set.xml', 'optimization_parameters__1.xml')

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_configuration_2(self):
        self.run_configuration(os.path.abspath(__file__), 'output__2', 'output_saved__2',
                               'model__2.xml', 'data_set.xml', 'optimization_parameters__2.xml')

    @unittest.skipIf(not torch.cuda.is_available(), 'cuda is not available')
    def test_configuration_3(self):
        self.run_configuration(os.path.abspath(__file__), 'output__3', 'output_saved__3',
                               'model__3.xml', 'data_set.xml', 'optimization_parameters__3.xml')

    def test_configuration_4(self):
        self.run_configuration(os.path.abspath(__file__), 'output__4', 'output_saved__4',
                               'model__4.xml', 'data_set.xml', 'optimization_parameters__4.xml')

    def test_configuration_5(self):
        self.run_configuration(os.path.abspath(__file__), 'output__5', 'output_saved__5',
                               'model__5.xml', 'data_set.xml', 'optimization_parameters__5.xml')

    def test_configuration_6(self):
        self.run_configuration(os.path.abspath(__file__), 'output__6', 'output_saved__6',
                               'model__6.xml', 'data_set.xml', 'optimization_parameters__6.xml')
