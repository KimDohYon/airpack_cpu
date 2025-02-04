#!/usr/bin/env python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

"""
Perform inference and benchmark inference performance using the ONNX Runtime.

Benchmarks are performed by repeatedly running inference on a random input
vector and measuring the total time taken.
"""

import os
import time
from typing import Optional, Union

import numpy as np
import onnxruntime


def onnx_bench(onnx_file: Union[str, os.PathLike],
               cplx_samples: int,
               batch_size: int = 128,
               num_inferences: Optional[int] = 100,
               input_dtype: np.number = np.float32) -> None:
    """
    Benchmarks a saved model using the ``onnxruntime`` inference engine.

    :param onnx_file:      Saved model file (``.onnx`` format)
    :param cplx_samples:   Input length of the neural network, in complex samples;
                           this is half of the ``input_length`` of the neural
                           network which operates on real values
    :param batch_size:     How many sets of ``cplx_samples`` inputs are batched
                           together in a single inference call
    :param num_inferences: Number of iterations to execute inference between
                           measurements of inference throughput
                           (if None, then run forever)
    :param input_dtype:    Data type of a single value (a single I or Q value,
                           not a complete complex (I, Q) sample): use one of
                           :class:`numpy.int16` or :class:`numpy.float32` here
    """

    # Setup the ONNX session
    sess = onnxruntime.InferenceSession(str(onnx_file))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    def infer_func(x):
        return sess.run([label_name], {input_name: x})[0]

    # Populate input buffer with test data
    buff_len = 2 * cplx_samples * batch_size
    sample_buffer = np.random.randn(buff_len).reshape((batch_size, 2 * cplx_samples)).astype(input_dtype)

    # Time the DNN Execution
    start_time = time.monotonic()
    for _ in range(num_inferences):
        infer_func(sample_buffer)
    elapsed_time = time.monotonic() - start_time
    total_cplx_samples = cplx_samples * batch_size * num_inferences

    throughput_msps = total_cplx_samples / elapsed_time / 1e6
    rate_gbps = throughput_msps * 2 * sample_buffer.itemsize * 8 / 1e3
    print('Result:')
    print('  Samples Processed : {:,}'.format(total_cplx_samples))
    print('  Processing Time   : {:0.3f} msec'.format(elapsed_time / 1e-3))
    print('  Throughput        : {:0.3f} MSPS'.format(throughput_msps))
    print('  Data Rate         : {:0.3f} Gbit / sec'.format(rate_gbps))
