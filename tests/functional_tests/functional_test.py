import _pickle as pickle
import logging
import os
import shutil
import subprocess
import unittest

import PIL.Image as pimg
import numpy as np

import deformetrica as dfca

logger = logging.getLogger(__name__)

DEFAULT_PRECISION = 1e-7


class FunctionalTest(unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.to_be_removed = []

    def run_configuration(self, path_to_test, output_folder, output_saved_folder, model_xml, data_set_xml, optimization_parameters_xml, command='estimate', precision=DEFAULT_PRECISION):
        # Run.
        path_to_deformetrica = os.path.normpath(os.path.join(path_to_test, '../../../../../../deformetrica/__main__.py'))
        path_to_model_xml = os.path.normpath(os.path.join(os.path.dirname(path_to_test), model_xml))
        path_to_optimization_parameters_xml = os.path.normpath(os.path.join(os.path.dirname(path_to_test), optimization_parameters_xml))
        path_to_data_set_xml = os.path.normpath(os.path.join(os.path.dirname(path_to_test), data_set_xml)) if data_set_xml is not None else None
        path_to_output = os.path.normpath(os.path.join(os.path.dirname(path_to_test), output_folder))
        # path_to_output = os.path.normpath(os.path.join(os.path.dirname(path_to_test), output_saved_folder))
        path_to_log = os.path.join(path_to_output, 'log.txt')
        if os.path.isdir(path_to_output):
            shutil.rmtree(path_to_output)
        os.mkdir(path_to_output)

        self.to_be_removed.append(path_to_output)

        # if data_set_xml is not None:
        #     cmd = 'if [ -f ~/.profile ]; then . ~/.profile; fi && ' \
        #           'bash -c \'conda activate deformetrica_env && python %s %s %s --dataset=%s --output=%s -v DEBUG > %s\'' % \
        #           (path_to_deformetrica, path_to_model_xml, path_to_optimization_parameters_xml, path_to_data_set_xml, path_to_output, path_to_log)
        # else:
        #     # without dataset
        #     cmd = 'if [ -f ~/.profile ]; then . ~/.profile; fi && ' \
        #           'bash -c \'conda activate deformetrica_env && python %s %s %s --output=%s -v DEBUG > %s\'' % \
        #           (path_to_deformetrica, path_to_model_xml, path_to_optimization_parameters_xml, path_to_output, path_to_log)
        if command in ['estimate', 'initialize']:
            cmd = 'if [ -f ~/.profile ]; then . ~/.profile; fi && ' \
                  '/bin/bash -c \'source ~/miniconda3/etc/profile.d/conda.sh && conda activate deformetrica_env && python %s %s %s %s --parameters=%s --output=%s -v DEBUG > %s\'' % \
                  (path_to_deformetrica, command, path_to_model_xml, path_to_data_set_xml, path_to_optimization_parameters_xml, path_to_output, path_to_log)
        elif command is 'compute':
            # without dataset
            cmd = 'if [ -f ~/.profile ]; then . ~/.profile; fi && ' \
                  '/bin/bash -c \'source ~/miniconda3/etc/profile.d/conda.sh && conda activate deformetrica_env && python %s compute %s --parameters=%s --output=%s -v DEBUG > %s\'' % \
                  (path_to_deformetrica, path_to_model_xml, path_to_optimization_parameters_xml, path_to_output, path_to_log)
        else:
            raise TypeError('command ' + command + ' was not recognized.')

        try:
            subprocess.check_call([cmd], shell=True)
        except subprocess.CalledProcessError as e:
            self.fail(e)

        # Initialize the comparison with saved results.
        path_to_output_saved = os.path.normpath(
            os.path.join(os.path.dirname(path_to_test), output_saved_folder))
        assert os.path.isdir(path_to_output_saved), 'No previously saved results: no point of comparison.'

        # If there is an available pickle dump, use it to conclude. Otherwise, extensively compare the output files.
        if command in ['estimate']:
            path_to_deformetrica_state = os.path.join(path_to_output, 'deformetrica-state.p')
            path_to_deformetrica_state_saved = os.path.join(path_to_output_saved, 'deformetrica-state.p')
            assert os.path.isfile(path_to_deformetrica_state_saved), 'The expected saved pickle dump file does not exist'
            assert os.path.isfile(path_to_deformetrica_state), 'The test did not produce the expected pickle dump file.'
            self._compare_pickle_dumps(path_to_deformetrica_state_saved, path_to_deformetrica_state,
                                       precision=precision)
        elif command in ['initialize']:
            path_to_deformetrica_state = os.path.join(path_to_output, '5_longitudinal_atlas_with_gradient_ascent', 'deformetrica-state.p')
            path_to_deformetrica_state_saved = os.path.join(path_to_output_saved, '5_longitudinal_atlas_with_gradient_ascent', 'deformetrica-state.p')
            assert os.path.isfile(path_to_deformetrica_state_saved), 'The expected saved pickle dump file does not exist'
            assert os.path.isfile(path_to_deformetrica_state), 'The test did not produce the expected pickle dump file.'
            self._compare_pickle_dumps(path_to_deformetrica_state_saved, path_to_deformetrica_state,
                                       precision=precision)
        elif command in ['compute']:
            self._compare_all_files(path_to_output_saved, path_to_output, precision=precision)

    def tearDown(self):
        if 'KEEP_OUTPUT' not in os.environ:
            for d in self.to_be_removed:
                shutil.rmtree(d)

            self.to_be_removed.clear()

        super().tearDown()

    ####################################################################################################################
    ### Utility methods:
    ####################################################################################################################

    def _compare_pickle_dumps(self, path_to_expected_deformetrica_state, path_to_actual_deformetrica_state, precision=DEFAULT_PRECISION):
        with open(path_to_expected_deformetrica_state, 'rb') as expected_deformetrica_state_file, \
                open(path_to_actual_deformetrica_state, 'rb') as actual_deformetrica_state_file:
            expected_deformetrica_state = pickle.load(expected_deformetrica_state_file)
            actual_deformetrica_state_saved = pickle.load(actual_deformetrica_state_file)
            self._assertStateEqual(expected_deformetrica_state, actual_deformetrica_state_saved, precision=precision)

    def _compare_all_files(self, path_to_expected_outputs, path_to_actual_outputs, precision=DEFAULT_PRECISION):
        expected_outputs = [f for f in os.listdir(path_to_expected_outputs) if not f.startswith('.') and not f.endswith('.log')]
        actual_outputs = [f for f in os.listdir(path_to_actual_outputs) if not f.startswith('.') and not f.endswith('.log')]
        self.assertEqual(len(expected_outputs), len(actual_outputs))

        for fn in expected_outputs:
            file_extension = os.path.splitext(fn)[1]
            path_to_expected_file = os.path.join(path_to_expected_outputs, fn)
            path_to_actual_file = os.path.join(path_to_actual_outputs, fn)
            self.assertTrue(os.path.isfile(path_to_actual_file))

            if fn in ['log.txt']:
                continue
            elif file_extension == '.txt':
                self._compare_txt_files(path_to_expected_file, path_to_actual_file, precision=precision)
            elif file_extension == '.vtk':
                self._compare_vtk_files(path_to_expected_file, path_to_actual_file, precision=precision)
            elif file_extension == '.png':
                self._compare_png_files(path_to_expected_file, path_to_actual_file)
            elif not file_extension == '':  # Case of the "log" file.
                msg = 'Un-checked file: %s. Please add the relevant comparison script for the file extensions "%s"' % \
                      (fn, file_extension)
                logger.warning(msg)

    def _assertStateEqual(self, expected, actual, precision=DEFAULT_PRECISION):
        if isinstance(expected, dict):
            self.assertTrue(isinstance(actual, dict))
            expected_keys = list(expected.keys())
            actual_keys = list(actual.keys())
            self.assertEqual(expected_keys, actual_keys)
            for key in expected_keys:
                self._assertStateEqual(expected[key], actual[key], precision=precision)

        elif isinstance(expected, np.ndarray):
            self.assertTrue(isinstance(actual, np.ndarray))
            self._compare_numpy_arrays(expected, actual, rtol=precision, atol=precision)

        else:
            self.assertEqual(expected, actual)

    def _compare_numpy_arrays(self, expected, actual, rtol=DEFAULT_PRECISION, atol=DEFAULT_PRECISION):
        self.assertTrue(np.allclose(expected, actual, rtol=rtol, atol=atol))

    def _compare_txt_files(self, path_to_expected_txt_file, path_to_actual_txt_file, precision=DEFAULT_PRECISION):
        expected = dfca.io.read_3D_array(path_to_expected_txt_file)
        actual = dfca.io.read_3D_array(path_to_actual_txt_file)
        self._compare_numpy_arrays(expected, actual, rtol=precision, atol=precision)

    def _compare_vtk_files(self, path_to_expected_vtk_file, path_to_actual_vtk_file, precision=DEFAULT_PRECISION):
        expected, expected_dimension = dfca.io.DeformableObjectReader.read_file(path_to_expected_vtk_file)
        actual, dimension = dfca.io.DeformableObjectReader.read_file(path_to_actual_vtk_file)
        self.assertEqual(expected_dimension, dimension)
        self._compare_numpy_arrays(expected, actual, rtol=precision, atol=precision)

    def _compare_png_files(self, path_to_expected_png_file, path_to_actual_png_file):
        expected = np.array(pimg.open(path_to_expected_png_file))
        actual = np.array(pimg.open(path_to_actual_png_file))
        mismatching_pixels_frequency = np.mean(np.abs((expected - actual) > 1e-2))
        self.assertTrue(mismatching_pixels_frequency < 0.005)

