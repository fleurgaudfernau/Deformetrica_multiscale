#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import gc

import pykeops
from api import Deformetrica
import support.utilities as utilities
from unit_tests import example_data_dir, sandbox_data_dir
import os
import time
import torch


data_dir = os.path.join(os.path.dirname(__file__), "data")

# # SMALL 3D MESH
# dataset_specifications = {
#     'dataset_filenames': [
#         [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala1.vtk',
#           'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo1.vtk'}],
#         # [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala2.vtk',
#         #   'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo2.vtk'}],
#         # [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala3.vtk',
#         #   'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo3.vtk'}],
#         # [{'amygdala': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amygdala4.vtk',
#         #   'hippo': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo4.vtk'}]
#         ],
#     # 'subject_ids': ['subj1', 'subj2', 'subj3', 'subj4']
#     'subject_ids': ['subj1']
# }
# template_specifications = {
#     'amygdala': {'deformable_object_type': 'SurfaceMesh',
#                  'kernel_type': 'keops', 'kernel_width': 15.0,
#                  'noise_std': 10.0,
#                  'filename': example_data_dir + '/atlas/landmark/3d/brain_structures/data/amyg_prototype.vtk',
#                  'attachment_type': 'varifold'},
#     'hippo': {'deformable_object_type': 'SurfaceMesh',
#               'kernel_type': 'keops', 'kernel_width': 15.0,
#               'noise_std': 6.0,
#               'filename': example_data_dir + '/atlas/landmark/3d/brain_structures/data/hippo_prototype.vtk',
#               'attachment_type': 'varifold'}
# }

# LARGE 3D mesh
dataset_specifications = {'dataset_filenames': [], 'subject_ids': []}
for file in os.listdir(data_dir + '/landmark/3d/right_hippocampus_2738'):
    subject_id, visit_age = utilities.adni_extract_from_file_name(file)

    dataset_specifications['dataset_filenames'].append(
        [{'hippo': data_dir + '/landmark/3d/right_hippocampus_2738/' + file}],
    )
    dataset_specifications['subject_ids'].append(subject_id)

template_specifications = {
    'hippo': {'deformable_object_type': 'SurfaceMesh',
              'kernel_type': 'keops', 'kernel_width': 15.0,
              'noise_std': 6.0,
              'filename': data_dir + '/landmark/3d/right_hippocampus_2738/sub-ADNI002S0729_ses-M00.vtk',
              'attachment_type': 'varifold'}
}

# # FULL T1 IMAGE
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


current_log_likelihood = 0.


def __estimator_callback(status_dict):
    global current_log_likelihood
    # current_iteration = status_dict['current_iteration']
    current_log_likelihood = status_dict['current_log_likelihood']
    return True


def deterministic_atlas_3d_brain_structure(kernel_type, nb_process, process_per_gpu, kernel_width):

    downsampling_factor = max(1, int(kernel_width / 2))
    logger.info('downsampling_factor=' + str(downsampling_factor))

    template_specifications['hippo']['kernel_type'] = kernel_type
    template_specifications['hippo']['kernel_width'] = kernel_width

    with Deformetrica(verbosity='DEBUG') as deformetrica:
        deformetrica.estimate_deterministic_atlas(
            template_specifications,
            dataset_specifications,
            estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 5,
                               'use_cuda': True, 'callback': __estimator_callback},
            model_options={'deformation_kernel_type': kernel_type, 'deformation_kernel_width': kernel_width, 'deformation_kernel_device': 'cuda',
                           'downsampling_factor': downsampling_factor,
                           'number_of_processes': nb_process, 'process_per_gpu': process_per_gpu},
            write_output=False)


def registration_3d_image(nb_process, number_of_time_points, kernel_width):

    downsampling_factor = max(1, int(kernel_width/2))
    logger.info('downsampling_factor=' + str(downsampling_factor))

    template_specifications['brain']['kernel_width'] = kernel_width

    with Deformetrica(verbosity='DEBUG') as deformetrica:
        deformetrica.estimate_registration(template_specifications, dataset_specifications,
                                           estimator_options={'optimization_method_type': 'GradientAscent', 'max_iterations': 20,
                                                              'use_cuda': False, 'callback': __estimator_callback},
                                           model_options={'deformation_kernel_type': 'keops', 'deformation_kernel_width': kernel_width,
                                                          'number_of_time_points': number_of_time_points, 'downsampling_factor': downsampling_factor,
                                                          'number_of_processes': nb_process, 'process_per_gpu': 1},
                                           write_output=False)


RUN_CONFIG = [
    # kernel_type, nb_process, process_per_gpu, kernel_width
    # (deterministic_atlas_3d_brain_structure, 'keops', 12, 1, 10.0),
    # (deterministic_atlas_3d_brain_structure, 'keops', 24, 1, 10.0),
    # (deterministic_atlas_3d_brain_structure, 'keops', 1, 1, 10.0),

    (deterministic_atlas_3d_brain_structure, 'keops', 2, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 3, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 4, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 6, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 8, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 10, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 12, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 16, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 20, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 24, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 28, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 32, 1, 10.0),
    (deterministic_atlas_3d_brain_structure, 'keops', 36, 1, 10.0),

    # nb_process, number_of_time_points, kernel_width
    # (registration_3d_image, 1, 3, 10.0),
    # (registration_3d_image, 1, 5, 10.0),
    # (registration_3d_image, 1, 7, 10.0),
    # (registration_3d_image, 1, 9, 10.0),
    # (registration_3d_image, 1, 11, 10.0),
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
        logger.info('current_log_likelihood: ' + str(current_log_likelihood))

        res_elapsed_time.append(elapsed_time)
        res_log_likelihood.append(current_log_likelihood)

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
