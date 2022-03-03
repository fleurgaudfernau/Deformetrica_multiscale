#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import gc

import pykeops
from api import Deformetrica
from core import default
from support import utilities
from unit_tests import example_data_dir, sandbox_data_dir
import os
import time
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)


data_dir = os.path.join(os.path.dirname(__file__), "data")


log_likelihoods = []


def __estimator_callback(status_dict):
    global log_likelihoods
    current_iteration = status_dict['current_iteration']
    if current_iteration == 1:
        log_likelihoods.append([])

    log_likelihoods[-1].append(status_dict['current_log_likelihood'])
    logger.info('>> log_likelihoods=' + str(log_likelihoods))
    return True


# # FULL T1 IMAGE (Registration)
# dataset_specifications = {
#     'dataset_filenames': [
#         # [{'brain': sandbox_data_dir + '/registration/image/3d/brains/data/s0021_7260_0.nii'}],
#         # [{'brain': sandbox_data_dir + '/registration/image/3d/brains/data/s0021_7960_8.nii'}],
#         [{'brain': sandbox_data_dir + '/registration/image/3d/brains/data/s0041_7110_0.nii'}]
#     ],
#     'subject_ids': ['s0021']
# }
# template_specifications = {
#     'brain': {'deformable_object_type': 'Image',
#               'kernel_type': 'keops', 'kernel_width': 10.0,
#               'noise_std': 1.0,
#               'filename': sandbox_data_dir + '/registration/image/3d/brains/data/s0021_7260_0.nii',
#               'attachment_type': 'varifold'}
# }
#
#
# current_log_likelihood = 0.
#
#
# def __estimator_callback(status_dict):
#     global current_log_likelihood
#     # current_iteration = status_dict['current_iteration']
#     current_log_likelihood = status_dict['current_log_likelihood']
#     return True
#
#
# def registration_3d_image(nb_process, number_of_time_points, kernel_width):
#
#     downsampling_factor = max(1, int(kernel_width/2))
#     logger.info('downsampling_factor=' + str(downsampling_factor))
#
#     template_specifications['brain']['kernel_width'] = kernel_width
#
#     with Deformetrica(verbosity='DEBUG') as deformetrica:
#         deformetrica.estimate_registration(template_specifications, dataset_specifications,
#                                            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 20,
#                                                               'use_cuda': False, 'callback': __estimator_callback},
#                                            model_options={'deformation_kernel_type': 'keops', 'deformation_kernel_width': kernel_width,
#                                                           'number_of_time_points': number_of_time_points, 'downsampling_factor': downsampling_factor,
#                                                           'number_of_processes': nb_process, 'process_per_gpu': 1},
#                                            write_output=False)





BASE_DIR = sandbox_data_dir + '/longitudinal_atlas/image/3d/hippocampi'


# dataset_specifications = {'subject_ids': set(), 'dataset_filenames': [], 'visit_ages': []}
dataset_specifications = {'subject_ids': set(), 'visit_ages': {}, 'dataset_filenames': {}}


for file in sorted(os.listdir(BASE_DIR + '/data')):
    if file.startswith("s") and file.endswith(".nii"):
        subject_id, visit_age, visit_id = utilities.longitudinal_extract_from_file_name(file)
        assert 0 < visit_age < 100, 'file is ' + file + ', subject_id=' + str(subject_id) + ', visit_age= ' + str(visit_age) + ', visit_id=' + str(visit_id)

        dataset_specifications['subject_ids'].add(subject_id)

        # subject_visit_ages.append(visit_age)
        # subject_visit_ids.append({'hippocampi': os.path.join(BASE_DIR, 'data', file)})

        # dataset_specifications['visit_ages'].append(subject_visit_ages)
        # dataset_specifications['dataset_filenames'].append(subject_visit_ids)

        if subject_id not in dataset_specifications['visit_ages']:
            dataset_specifications['visit_ages'][subject_id] = []
        dataset_specifications['visit_ages'][subject_id].append(visit_age)

        if subject_id not in dataset_specifications['dataset_filenames']:
            dataset_specifications['dataset_filenames'][subject_id] = []
        dataset_specifications['dataset_filenames'][subject_id].append({'hippocampi': os.path.join(BASE_DIR, 'data', file)})


# convert from dict to list
dataset_specifications['subject_ids'] = sorted(list(dataset_specifications['subject_ids']))
dataset_specifications['visit_ages'] = list(dataset_specifications['visit_ages'].values())
dataset_specifications['dataset_filenames'] = list(dataset_specifications['dataset_filenames'].values())


template_specifications = {
    'hippocampi': {'deformable_object_type': 'Image',
                   'noise_std': 0.0997,
                   # 'kernel_type': 'keops', 'kernel_width': 10.0,
                   'filename': os.path.join(BASE_DIR, 'data', 'ForInitialization__Template_right_hippocampus__FromLongitudinalAtlas.nii'),
                   'noise_variance_prior_normalized_dof': 0.01,
                   'noise_variance_prior_scale_std': 1.
                   }
}


def longitudinal_atlas_3d_image(nb_process, max_iterations=2, max_line_search_iterations=5):
    kernel_width = 10.0
    # number_of_time_points = 11
    number_of_time_points = 6
    # concentration_of_time_points = 5
    concentration_of_time_points = 2
    # downsampling_factor = 5
    downsampling_factor = 1

    # default.update_dtype('float32')

    # downsampling_factor = max(1, int(kernel_width/2))
    # logger.info('downsampling_factor=' + str(downsampling_factor))

    logger.info('============================================================')
    logger.info('nb_process=' + str(nb_process))
    logger.info('max_iterations=' + str(max_iterations))
    logger.info('max_line_search_iterations=' + str(max_line_search_iterations))
    logger.info('kernel_width=' + str(kernel_width))
    logger.info('number_of_time_points=' + str(number_of_time_points))
    logger.info('concentration_of_time_points=' + str(concentration_of_time_points))
    logger.info('downsampling_factor=' + str(downsampling_factor))
    logger.info('============================================================')

    template_specifications['hippocampi']['kernel_width'] = kernel_width

    with Deformetrica(verbosity='DEBUG') as deformetrica:
        torch.manual_seed(42)
        np.random.seed(42)

        deformetrica.estimate_longitudinal_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'McmcSaem', 'initial_step_size': 1e-4,
                               'convergence_tolerance': 1e-4, 'max_iterations': max_iterations,
                               'max_line_search_iterations': max_line_search_iterations, 'sample_every_n_mcmc_iters': 25, 'save_every_n_iters': 1000,
                               'callback': __estimator_callback},
            model_options={'deformation_kernel_type': 'keops', 'deformation_kernel_width': kernel_width, 'downsampling_factor': downsampling_factor,
                           'concentration_of_time_points': concentration_of_time_points, 'number_of_time_points': number_of_time_points, 't0': 72.1944,

                           'initial_control_points': os.path.join(BASE_DIR, 'data', 'ForInitialization__ControlPoints__FromLongitudinalAtlas.txt'),
                           'initial_momenta': os.path.join(BASE_DIR, 'data', 'ForInitialization__Momenta__FromLongitudinalAtlas.txt'),
                           'initial_modulation_matrix': os.path.join(BASE_DIR, 'data', 'ForInitialization__ModulationMatrix__FromLongitudinalAtlas.txt'),
                           # 'initial_onset_ages': os.path.join(BASE_DIR, 'data', 'ForInitialization__OnsetAges__FromLongitudinalAtlas.txt'),
                           # 'initial_acceleration': os.path.join(BASE_DIR, 'data', 'ForInitialization__LogAccelerations__FromLongitudinalAtlas.txt'),
                           # 'initial_sources': os.path.join(BASE_DIR, 'data', 'ForInitialization__Sources__FromLongitudinalAtlas.txt'),
                           'initial_time_shift_variance': 1.1749 ** 2,
                           'initial_acceleration_variance': 1.33 ** 2,
                           'number_of_sources': len(dataset_specifications['subject_ids']),

                           'number_of_processes': nb_process, 'process_per_gpu': 1},
            write_output=False)


RUN_CONFIG = [
    # nb_process, number_of_time_points, kernel_width
    # (registration_3d_image, 1, 3, 10.0),
    # (registration_3d_image, 1, 5, 10.0),
    # (registration_3d_image, 1, 7, 10.0),
    # (registration_3d_image, 1, 9, 10.0),
    # (registration_3d_image, 1, 11, 10.0),

    # nb_process, max_iterations=2, max_line_search_iterations=5
    # (longitudinal_atlas_3d_image, 1, 1, 1),    # warmup for keops compilation
    # (longitudinal_atlas_3d_image, 1, 1),
    (longitudinal_atlas_3d_image, 2, 1),
    # (longitudinal_atlas_3d_image, 3),
    # (longitudinal_atlas_3d_image, 4, 200),
    # (longitudinal_atlas_3d_image, 8),
    # (longitudinal_atlas_3d_image, 16),
    # (longitudinal_atlas_3d_image, 20),
    # (longitudinal_atlas_3d_image, 24),
    # (longitudinal_atlas_3d_image, 32),
    # (longitudinal_atlas_3d_image, 36),
]


if __name__ == "__main__":
    logger.info('torch.__version__=' + torch.__version__)
    logger.info('pykeops.__version__=' + pykeops.__version__)

    res_elapsed_time = []
    res_log_likelihood = []

    for current_run_config in RUN_CONFIG:
        func, *args = current_run_config
        logger.info('>>>>>>>>>>>>> func=' + str(func) + ', args=' + str(args))

        start = time.perf_counter()
        func(*args)
        elapsed_time = time.perf_counter()-start
        logger.info('elapsed_time: ' + str(elapsed_time))
        logger.info('log_likelihoods: ' + str(log_likelihoods))

        res_elapsed_time.append(elapsed_time)
        res_log_likelihood.append(log_likelihoods)

        log_likelihoods = []

        # cleanup between runs
        time.sleep(0.5)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.5)

    logger.info('===== RESULTS =====')
    logger.info(res_elapsed_time)
    logger.info(res_log_likelihood)

    # assert len(nb_processes) == len(results)
    #
    # index = np.arange(len(RUN_CONFIG))
    # bar_width = 0.2
    # opacity = 0.4
    #
    # fig, ax = plt.subplots()
    #
    # ax.bar(index + bar_width, results, bar_width, label=':')
    #
    # ax.set_xlabel('Nb process')
    # ax.set_ylabel('Runtime (s)')
    # ax.set_title('Runtime by number of processes')
    # # ax.set_xticks(index + bar_width * ((len(kernels)*len(initial_devices))/2) - bar_width/2)
    # # ax.set_xticklabels([r['setup']['tensor_size'] for r in results if r['setup']['device'] == 'cpu'])
    # ax.legend()
    #
    # fig.tight_layout()
    #
    # plt.show()
