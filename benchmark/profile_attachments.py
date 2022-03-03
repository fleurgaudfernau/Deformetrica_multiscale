#!/usr/bin/env python
# -*- encoding: utf-8 -*-


"""

ShapeMI at MICCAI 2018
https://shapemi.github.io/


Benchmark CPU vs GPU on small (500 points) and large (5000 points) meshes.

"""

import gc
import itertools
import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch

import support.kernels as kernel_factory
from benchmark.memory_profile_tool import start_memory_profile, stop_and_clear_memory_profile
from core import default
from core.model_tools.attachments.multi_object_attachment import MultiObjectAttachment
from core.observations.deformable_objects.landmarks.surface_mesh import SurfaceMesh
from in_out.deformable_object_reader import DeformableObjectReader

path_to_small_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_500_cells_1.vtk'
path_to_small_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_500_cells_2.vtk'
path_to_large_surface_mesh_1 = 'data/landmark/surface_mesh/hippocampus_5000_cells_1.vtk'
path_to_large_surface_mesh_2 = 'data/landmark/surface_mesh/hippocampus_5000_cells_2.vtk'


class ProfileAttachments:
    def __init__(self, kernel_type, kernel_device='CPU', use_cuda=False, data_size='small'):

        np.random.seed(42)
        kernel_width = 10.
        tensor_scalar_type = default.tensor_scalar_type

        if kernel_device.upper() == 'CPU':
            tensor_scalar_type = torch.FloatTensor
        elif kernel_device.upper() == 'GPU':
            tensor_scalar_type = torch.cuda.FloatTensor
        else:
            raise RuntimeError

        self.multi_object_attachment = MultiObjectAttachment(['varifold'], [kernel_factory.factory(kernel_type, kernel_width, device=kernel_device)])

        self.kernel = kernel_factory.factory(kernel_type, kernel_width, device=kernel_device)

        reader = DeformableObjectReader()

        if data_size == 'small':
            self.surface_mesh_1 = reader.create_object(path_to_small_surface_mesh_1, 'SurfaceMesh', tensor_scalar_type)
            self.surface_mesh_2 = reader.create_object(path_to_small_surface_mesh_2, 'SurfaceMesh', tensor_scalar_type)
            self.surface_mesh_1_points = tensor_scalar_type(self.surface_mesh_1.get_points())
        elif data_size == 'large':
            self.surface_mesh_1 = reader.create_object(path_to_large_surface_mesh_1, 'SurfaceMesh', tensor_scalar_type)
            self.surface_mesh_2 = reader.create_object(path_to_large_surface_mesh_2, 'SurfaceMesh', tensor_scalar_type)
            self.surface_mesh_1_points = tensor_scalar_type(self.surface_mesh_1.get_points())
        else:
            data_size = int(data_size)
            connectivity = np.array(list(itertools.combinations(range(100), 3))[:data_size])  # up to ~16k.
            self.surface_mesh_1 = SurfaceMesh(3)
            self.surface_mesh_1.set_points(np.random.randn(np.max(connectivity) + 1, 3))
            self.surface_mesh_1.set_connectivity(connectivity)
            self.surface_mesh_1.update()
            self.surface_mesh_2 = SurfaceMesh(3)
            self.surface_mesh_2.set_points(np.random.randn(np.max(connectivity) + 1, 3))
            self.surface_mesh_2.set_connectivity(connectivity)
            self.surface_mesh_2.update()
            self.surface_mesh_1_points = tensor_scalar_type(self.surface_mesh_1.get_points())

    def current_attachment(self):
        return self.multi_object_attachment.current_distance(
            self.surface_mesh_1_points, self.surface_mesh_1, self.surface_mesh_2, self.kernel)

    def varifold_attachment(self):
        return self.multi_object_attachment.varifold_distance(
            self.surface_mesh_1_points, self.surface_mesh_1, self.surface_mesh_2, self.kernel)

    def current_attachment_with_backward(self):
        self.surface_mesh_1_points.requires_grad_(True)
        attachment = self.current_attachment()
        attachment.backward()

    def varifold_attachment_with_backward(self):
        self.surface_mesh_1_points.requires_grad_(True)
        attachment = self.varifold_attachment()
        attachment.backward()


class BenchRunner:
    def __init__(self, kernel, method_to_run):
        self.obj = ProfileAttachments(kernel[0], kernel[1], kernel[2], method_to_run[0])
        self.to_run = getattr(self.obj, method_to_run[1])

        # Activate the garbage collector.
        gc.enable()

        # run once for warm-up: cuda pre-compile with keops
        self.run()
        # logger.info('BenchRunner::__init()__ done')

    """ The method that is to be benched must reside within the run() method """
    def run(self):
        self.to_run()

        logger.info('.', end='')    # uncomment to show progression

    def __exit__(self):
        logger.info('BenchRunner::__exit()__')


def build_setup():
    kernels = []
    method_to_run = []

    # Small sizes.
    for data_size in ['100', '200', '400', '800', '1600', '3200', '6400']:
        for attachment_type in ['varifold', 'current']:
            for kernel_type in [('keops', 'CPU', False), ('keops', 'GPU', False), ('keops', 'GPU', True),
                                ('torch', 'CPU', False), ('torch', 'GPU', False), ('torch', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((data_size, attachment_type + '_attachment_with_backward'))

    # Large sizes.
    for data_size in ['12800']:
        for attachment_type in ['varifold', 'current']:
            for kernel_type in [('keops', 'CPU', False), ('keops', 'GPU', False), ('keops', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((data_size, attachment_type + '_attachment_with_backward'))

    # Very large sizes.
    for data_size in ['25600', '51200']:
        for attachment_type in ['varifold', 'current']:
            for kernel_type in [('keops', 'GPU', False), ('keops', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((data_size, attachment_type + '_attachment_with_backward'))

    # Huge sizes.
    for data_size in ['102400', '204800']:
        for attachment_type in ['varifold', 'current']:
            for kernel_type in [('keops', 'GPU', False), ('keops', 'GPU', True)]:
                kernels.append(kernel_type)
                method_to_run.append((data_size, attachment_type + '_attachment_with_backward'))

    # kernels = [('torch', 'CPU', False)]
    # method_to_run = [('50', 'varifold_attachment_with_backward')]

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
        np.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'results_profile_attachments.npy'),
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

                extracted_data = [r['max'] for r in results if r['setup']['kernel'] == k]

                assert(len(extracted_data) > 0)
                assert(len(extracted_data) == len(index))

                ax.bar(index + bar_width * i, extracted_data, bar_width, alpha=opacity, label=k[0] + ':' + k[1])
                i = i+1

            # bar1 = ax.bar(index, cpu_res, bar_width, alpha=0.4, color='b', label='cpu')
            # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=0.4, color='g', label='cuda')

            ax.set_xlabel('TODO')
            ax.set_ylabel('Runtime (s)')
            ax.set_title('TODO')
            ax.set_xticks(index + bar_width * ((len(kernels))/2) - bar_width/2)
            ax.set_xticklabels([r['setup']['method_to_run'][1] for r in results])
            ax.legend()

            # for tick in ax.get_xticklabels():
            #     tick.set_rotation(45)

            fig.tight_layout()

            plt.show()
