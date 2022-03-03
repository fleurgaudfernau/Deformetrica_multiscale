#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import support.kernels as kernel_factory
import torch


class BenchRunner:
    def __init__(self, kernel, tensor_size, tensor_initial_device='cpu'):
        # tensor_size = (4, 3)
        # logger.info('BenchRunner::__init()__ getting kernel and initializing tensors with size ' + str(tensor_size))

        torch.manual_seed(42)

        self.kernel_instance = kernel_factory.factory(kernel, kernel_width=1.)

        self.x = torch.rand(tensor_size, device=torch.device(tensor_initial_device))
        self.y = torch.rand(tensor_size, device=torch.device(tensor_initial_device))
        self.p = torch.ones(tensor_size, device=torch.device(tensor_initial_device))

        # run once for warm-up: cuda pre-compile
        self.res = self.kernel_instance.convolve(self.x, self.y, self.p)
        # logger.info('BenchRunner::__init()__ done')

    def run(self):
        self.res = self.kernel_instance.convolve(self.x, self.y, self.p)

        # logger.info(self.res)
        # move to CPU
        # self.res.to(torch.device('cpu'))

        # self.res = None
        # torch.cuda.empty_cache()
        logger.info('.', end='')

    def __exit__(self):
        logger.info('BenchRunner::__exit()__')


def build_setup():
    # kernels = ['keops']
    kernels = ['torch']
    # initial_devices = ['cuda:0']
    initial_devices = ['cpu']
    tensor_sizes = [(4, 3), (16, 3), (32, 3), (64, 3), (128, 3), (256, 3)]
    # tensor_sizes = [(64, 3), (128, 3), (256, 3), (512, 3)]
    setups = []

    for k, d, t in [(k, d, t) for k in kernels for d in initial_devices for t in tensor_sizes]:
        bench_setup = '''
from __main__ import BenchRunner
bench = BenchRunner('{kernel}', {tensor}, '{device}')
'''.format(kernel=k, tensor=str(t), device=d)

        setups.append({'kernel': k, 'device': d, 'tensor_size': t, 'bench_setup': bench_setup})
    return setups, kernels, initial_devices, len(tensor_sizes)


if __name__ == "__main__":
    import timeit

    results = []

    build_setup, kernels, initial_devices, tensor_size_len = build_setup()

    # cudaprofile.start()

    # prepare and run bench
    for setup in build_setup:
        logger.info('running setup ' + str(setup))

        res = {}
        res['setup'] = setup
        res['data'] = timeit.repeat("bench.run()", number=1, repeat=1, setup=setup['bench_setup'])
        res['min'] = min(res['data'])
        res['max'] = max(res['data'])

        logger.info('')
        logger.info(res)
        results.append(res)

    # cudaprofile.stop()

    # logger.info('cpu: ' + str(timeit.repeat("bench.run()", number=50000, repeat=3, setup=setup_cpu)))
    # logger.info('cuda: ' + str(timeit.repeat("bench.run()", number=50000, repeat=3, setup=setup_cuda)))
    # cpu_res = [r['max'] for r in results if r['setup']['device'] == 'cpu']
    # cuda_res = [r['max'] for r in results if r['setup']['device'] == 'cuda:0']
    # assert(len(cpu_res) == len(cuda_res))

    fig, ax = plt.subplots()
    # plt.ylim(ymin=0)
    # ax.set_yscale('log')

    index = np.arange(tensor_size_len)
    bar_width = 0.2
    opacity = 0.4

    # extract data from raw data and add to plot
    i = 0
    for d, k in [(d, k) for d in initial_devices for k in kernels]:
        extracted_data = [r['max'] for r in results if r['setup']['device'] == d if r['setup']['kernel'] == k]
        assert(len(extracted_data) == len(index))

        ax.bar(index + bar_width * i, extracted_data, bar_width, alpha=opacity, label=d + ':' + k)
        i = i+1

    # bar1 = ax.bar(index, cpu_res, bar_width, alpha=0.4, color='b', label='cpu')
    # bar2 = ax.bar(index + bar_width, cuda_res, bar_width, alpha=0.4, color='g', label='cuda')

    ax.set_xlabel('Tensor size')
    ax.set_ylabel('Runtime (s)')
    ax.set_title('Runtime by device/size')
    ax.set_xticks(index + bar_width * ((len(kernels)*len(initial_devices))/2) - bar_width/2)
    ax.set_xticklabels([r['setup']['tensor_size'] for r in results if r['setup']['device'] == 'cpu'])
    ax.legend()

    fig.tight_layout()

    plt.show()
