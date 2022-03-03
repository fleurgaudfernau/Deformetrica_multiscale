#!/usr/bin/env python
# -*- encoding: utf-8 -*-


"""

ShapeMI at MICCAI 2018
https://shapemi.github.io/


Benchmark CPU vs GPU on small (500 points) and large (5000 points) meshes.

"""

import os
import sys
import warnings

import gc
import matplotlib.pyplot as plt
import numpy as np
import support.kernels as kernel_factory
import torch
import itertools

from memory_profile_tool import start_memory_profile, stop_and_clear_memory_profile
from in_out.deformable_object_reader import DeformableObjectReader
from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from core.observations.deformable_objects.deformable_multi_object import DeformableMultiObject
from support.utilities.general_settings import Settings
from core.models.model_functions import create_regular_grid_of_points, remove_useless_control_points
from core.model_tools.deformations.exponential import Exponential
from core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from core.observations.deformable_objects.image import Image

path_to_small_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_500_cells_1.vtk'
path_to_small_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_500_cells_2.vtk'
path_to_large_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_5000_cells_1.vtk'
path_to_large_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_5000_cells_2.vtk'


class ProfileDeformations:
    def __init__(self, kernel_type, kernel_device='CPU', use_cuda=False, data_type='landmark', data_size='small'):

        np.random.seed(42)
        kernel_width = 10.

        if use_cuda:
            self.tensor_scalar_type = torch.cuda.FloatTensor
        else:
            self.tensor_scalar_type = torch.FloatTensor

        self.exponential = Exponential(kernel=kernel_factory.factory(kernel_type, kernel_width, kernel_device),
                                       number_of_time_points=11, use_rk2_for_flow=False, use_rk2_for_shoot=False)

        if data_type.lower() == 'landmark':
            reader = DeformableObjectReader()
            if data_size == 'small':
                surface_mesh = reader.create_object(path_to_small_surface_mesh_1, 'SurfaceMesh')
                self.control_points = create_regular_grid_of_points(surface_mesh.bounding_box, kernel_width, surface_mesh.dimension)
            elif data_size == 'large':
                surface_mesh = reader.create_object(path_to_large_surface_mesh_1, 'SurfaceMesh')
                self.control_points = create_regular_grid_of_points(surface_mesh.bounding_box, kernel_width, surface_mesh.dimension)
            else:
                connectivity = np.array(list(itertools.combinations(range(100), 3))[:int(data_size)])  # up to ~16k.
                surface_mesh = SurfaceMesh(3)
                surface_mesh.set_points(np.random.randn(np.max(connectivity) + 1, surface_mesh.dimension))
                surface_mesh.set_connectivity(connectivity)
                surface_mesh.update()
                self.control_points = np.random.randn(int(data_size) // 10, 3)
            # self.template.object_list.append(surface_mesh)
            self.template = DeformableMultiObject([surface_mesh])

        elif data_type.lower() == 'image':
            image = Image(3)
            image.set_intensities(np.random.randn(int(data_size), int(data_size), int(data_size)))
            image.set_affine(np.eye(4))
            image.downsampling_factor = 5.
            image.update()
            self.control_points = create_regular_grid_of_points(image.bounding_box, kernel_width, image.dimension)
            self.control_points = remove_useless_control_points(self.control_points, image, kernel_width)
            # self.template.object_list.append(image)
            self.template = DeformableMultiObject([image])

        else:
            raise RuntimeError('Unknown data_type argument. Choose between "landmark" or "image".')

        # self.template.update()
        self.momenta = np.random.randn(*self.control_points.shape)

    def forward(self):
        self.exponential.set_initial_template_points(
            {key: self.tensor_scalar_type(value) for key, value in self.template.get_points().items()})
        self.exponential.set_initial_control_points(self.tensor_scalar_type(self.control_points))
        self.exponential.set_initial_momenta(self.tensor_scalar_type(self.momenta))
        self.exponential.update()
        deformed_points = self.exponential.get_template_points()
        deformed_data = self.template.get_deformed_data(
            deformed_points,
            {key: self.tensor_scalar_type(value) for key, value in self.template.get_data().items()})

    def forward_and_backward(self):
        self.exponential.set_initial_template_points({key: self.tensor_scalar_type(value).requires_grad_(True)
                                                      for key, value in self.template.get_points().items()})
        self.exponential.set_initial_control_points(self.tensor_scalar_type(self.control_points).requires_grad_(True))
        self.exponential.set_initial_momenta(self.tensor_scalar_type(self.momenta).requires_grad_(True))
        self.exponential.update()
        deformed_points = self.exponential.get_template_points()
        deformed_data = self.template.get_deformed_data(deformed_points, {key: self.tensor_scalar_type(value) for key, value in self.template.get_data().items()})
        for key, value in deformed_data.items():
            value.backward(torch.ones(value.size()).type(self.tensor_scalar_type))


class BenchRunner:
    def __init__(self, kernel, method_to_run):
        self.obj = ProfileDeformations(kernel[0], kernel[1], kernel[2], method_to_run[0], method_to_run[1])
        self.to_run = getattr(self.obj, method_to_run[2])

        # Activate the garbage collector.
        gc.enable()

        # run once for warm-up: cuda pre-compile with keops
        self.run()
        # logger.info('BenchRunner::__init()__ done')

    """ The method that is to be benched must reside within the run() method """

    def run(self):
        self.to_run()

        logger.info('.', end='')  # uncomment to show progression

    def __exit__(self):
        logger.info('BenchRunner::__exit()__')


def build_setup():
    kernels = []
    method_to_run = []

    # Small sizes.
    for object_type in ['landmark', 'image']:
        for data_size in {'landmark': ['100', '200', '400', '800', '1600', '3200', '6400'],
                          'image': ['16', '32', '64', '96']}[object_type]:
            for kernel_type in [('keops', 'CPU', False), ('keops', 'GPU', False), ('keops', 'GPU', True),
                                ('torch', 'CPU', False), ('torch', 'GPU', False), ('torch', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((object_type, data_size, 'forward_and_backward'))

    # Large sizes.
    for object_type in ['landmark', 'image']:
        for data_size in {'landmark': ['12800', '25600'],
                          'image': ['128', '160']}[object_type]:
            for kernel_type in [('keops', 'CPU', False), ('keops', 'GPU', False), ('keops', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((object_type, data_size, 'forward_and_backward'))

    # Very large sizes.
    for object_type in ['landmark', 'image']:
        for data_size in {'landmark': ['51200'],
                          'image': ['192', '256']}[object_type]:
            for kernel_type in [('keops', 'GPU', False), ('keops', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((object_type, data_size, 'forward_and_backward'))

    # Huge sizes.
    for object_type in ['landmark']:
        for data_size in {'landmark': ['102400', '204800']}[object_type]:
            for kernel_type in [('keops', 'GPU', False), ('keops', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((object_type, data_size, 'forward_and_backward'))

    # kernels = [('torch', 'CPU', False)]
    # method_to_run = [('image', '16', 'forward_and_backward')]

    setups = []
    for (k, m) in zip(kernels, method_to_run):
        bench_setup = '''
from __main__ import BenchRunner
import torch
bench = BenchRunner({kernel}, {method_to_run})
'''.format(kernel=k, method_to_run=m)

        setups.append({'kernel': k, 'method_to_run': m, 'bench_setup': bench_setup})
    return setups, kernels, method_to_run


if __name__ == "__main__":
    import timeit

    results = []

    build_setup, kernels, method_to_run = build_setup()
    number = 100

    # prepare and run bench
    for setup in build_setup:
        logger.info('running setup ' + str(setup))

        res = {}
        res['setup'] = setup
        memory_profiler = start_memory_profile()
        res['data'] = [elt / float(number)
                       for elt in timeit.repeat("bench.run()", number=number, repeat=1, setup=setup['bench_setup'])]
        res['memory_profile'] = stop_and_clear_memory_profile(memory_profiler)
        res['min'] = min(res['data'])
        res['max'] = max(res['data'])
        res['mean'] = sum(res['data']) / float(len(res['data']))

        logger.info('')
        logger.info(res['data'])
        results.append(res)

        # Dump the results.
        np.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results_profile_deformations.npy'),
                np.array(results))

    # Optionally make a plot.
    if len(sys.argv) > 1:
        if not sys.argv[1] == '--plot':
            msg = 'Unknown command-line option: "%s". Ignoring.' % sys.argv[1]
            warnings.warn(msg)
        else:
            fig, ax = plt.subplots()
            # plt.ylim(ymin=0)
            # ax.set_yscale('log')

            index = np.arange(len(method_to_run))
            bar_width = 0.2
            opacity = 0.4

            # extract data from raw data and add to plot
            i = 0
            for k in [(k) for k in kernels]:
                extracted_data = [r['max'] for r in results
                                  if r['setup']['kernel'] == k]

                assert (len(extracted_data) > 0)
                assert (len(extracted_data) == len(index))

                ax.bar(index + bar_width * i, extracted_data, bar_width, alpha=opacity, label=k[0] + ':' + k[1])
                i = i + 1

            # bar1 = ax.bar(index, cpu_res, bar_width, alpha=0.4, color='b', label='cpu')
            # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=0.4, color='g', label='cuda')

            ax.set_xlabel('TODO')
            ax.set_ylabel('Runtime (s)')
            ax.set_title('TODO')
            ax.set_xticks(index + bar_width * ((len(kernels)) / 2) - bar_width / 2)
            ax.set_xticklabels([r['setup']['method_to_run'][1] for r in results])
            ax.legend()

            # for tick in ax.get_xticklabels():
            #     tick.set_rotation(45)

            fig.tight_layout()

            plt.show()
