#!/usr/bin/env python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

"""
Utility code for performing inference using the TensorRT framework with
settings optimized for AIR-T hardware.
"""

import atexit

import numpy as np
import pycuda.driver as cuda


def make_cuda_context(gpu_index: int = 0) -> cuda.Context:
    """Initializes a CUDA context for use with the selected GPU and makes it active.

    This context is created with a set of flags that will allow us to use
    device mapped (pinned) memory that supports zero-copy operations on the AIR-T.

    :param gpu_index: Which GPU in the system to use, defaults to the first GPU (index 0)
    """
    cuda.init()
    ctx_flags = (cuda.ctx_flags.SCHED_AUTO | cuda.ctx_flags.MAP_HOST)
    cuda_context = cuda.Device(gpu_index).make_context(ctx_flags)
    atexit.register(cuda_context.pop)  # ensure context is cleaned up


class MappedBuffer:
    """A device-mapped memory buffer for sharing data between CPU and GPU.

    Once created, the :attr:`host` field can be used to access the memory
    from CPU as a :class:`numpy.ndarray`, and the :attr:`device` field
    can be used to access the memory from the GPU.

    Example usage:

    .. code-block:: python

        # Create a buffer of 16 single-precision floats
        buffer = MappedBuffer(num_elems=16, dtype=numpy.float32)
        # Zero the buffer by writing to it on CPU
        buffer.host[:] = 0.0
        # Pass the device pointer to an API that works with GPU buffers
        func_that_uses_gpu_buffer(buffer.device)

    .. note::

        Device-mapped memory is meant for Jetson embedded GPUs like the one
        found on the AIR-T, where both the host and device pointers refer to
        the same physical memory. Using this type of memory buffer on desktop
        GPUs will be very slow.
    """

    _MEM_FLAGS = cuda.host_alloc_flags.DEVICEMAP

    def __init__(self, num_elems: int, dtype: np.number) -> None:
        """
        :param int num_elems:     Number of elements in the created buffer
        :param numpy.dtype dtype: Data type of an element (e.g.,
                                  :class:`numpy.float32` or :class:`numpy.int16`)
        :var numpy.ndarray host:  Access to the buffer from the CPU
        :var CUdeviceptr device:  Access to the buffer from the GPU
        """
        self.host = cuda.pagelocked_empty(num_elems, dtype,
                                          mem_flags=MappedBuffer._MEM_FLAGS)
        self.device = self.host.base.get_device_pointer()
