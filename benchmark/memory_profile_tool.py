import resource
import sys
import time
from threading import Thread
from memory_profiler import memory_usage

import GPUtil
import torch


# _cudart = ctypes.CDLL('libcudart.so')
#
#
# def start_cuda_profile():
#     # As shown at http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__PROFILER.html,
#     # the return value will unconditionally be 0. This check is just in case it changes in
#     # the future.
#     ret = _cudart.cudaProfilerStart()
#     if ret != 0:
#         raise Exception("cudaProfilerStart() returned %d" % ret)
#
#
# def stop_cuda_profile():
#     ret = _cudart.cudaProfilerStop()
#     if ret != 0:
#         raise Exception("cudaProfilerStop() returned %d" % ret)


class MemoryProfiler(Thread):
    def __init__(self, freq=0.1):
        Thread.__init__(self)
        self.freq = freq
        self.run_flag = True
        self.data = {'ram': []}

    def run(self):
        # logger.info('MemoryProfiler::run()')
        while self.run_flag:
            self.data['ram'].append(self.current_ram_usage())
            time.sleep(self.freq)

    def stop(self):
        # logger.info('MemoryProfiler::stop()')
        self.run_flag = False
        self.join()
        return dict(self.data)

    def clear(self):
        self.data.clear()

    @staticmethod
    def current_ram_usage():
        return memory_usage(-1, interval=0)[0]    # -1 is for current process


def start_memory_profile(freq=0.001):
    ret = MemoryProfiler(freq)
    ret.start()
    return ret


def stop_memory_profile(memory_profiler):
    return memory_profiler.stop()


def stop_and_clear_memory_profile(memory_profiler):
    ret = memory_profiler.stop()
    clear_memory_profile(memory_profiler)
    return ret


def clear_memory_profile(memory_profiler):
    memory_profiler.clear()



