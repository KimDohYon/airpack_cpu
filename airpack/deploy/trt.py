#!/usr/bin/env python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

"""
This package provides functionality to optimize a neural network using NVIDIA's
TensorRT framework and perform inference using optimized models.

Optimized models are saved in ``.plan`` format, which is an internal,
platform-specific data format for TensorRT. Since TensorRT optimization
functions by running many variations of the network on the target hardware,
it must be executed on the platform that will be used for inference,
i.e., the AIR-T for final deployment.

The basic workflow is as follows:

1. Save your trained model to an ONNX file
2. Optimize the model using TensorRT with the :func:`onnx2plan` function.
3. Create a :class:`TrtInferFromPlan` object for your optimized model and
   use it to perform inference.
"""

import os
import pathlib
import time
import warnings

import numpy as np
from typing import Union, Optional

import tensorrt as trt


# Default top-level inference settings
from airpack.deploy import trt_utils
from airpack.deploy.trt_utils import MappedBuffer

_LOGGER = trt.Logger(trt.Logger.VERBOSE)
"""Default TensorRT logging object"""


def onnx2plan(onnx_file: Union[str, os.PathLike],
              input_node_name: str = 'input',
              input_port_name: str = '',
              input_len: int = 4096,
              fp16_mode: bool = True,
              max_workspace_size: int = 1073741824,
              max_batch_size: int = 128,
              verbose: bool = False
              ) -> pathlib.Path:
    """Optimize the provided ONNX model using TensorRT and save the result.

    The optimized model will have a ``.plan`` extension and be saved in the same
    folder as the input ONNX model.

    :param onnx_file:
        Filename of the ONNX model to optimize
    :param input_node_name:
        Name of the ONNX model's input layer
    :param input_port_name:
    :param input_len:
        Length of the ONNX model's input layer, determined when the model was created
    :param fp16_mode:
        Try to use reduced precision (float16) layers if performace would improve
    :param max_workspace_size:
        Maximum scratch memory that the TensorRT optimizer may use, defaults to 1GB.
        The default value can be used in most situations and may only need to be
        reduced if using very low-end GPU hardware
    :param max_batch_size:
        The maximum batch size to optimize for. When running inference using the
        optimized model, the chosen batch size must be less than the maximum
        specified here
    :param verbose:
        Print extra information about the optimized model
    """

    onnx_file = pathlib.Path(onnx_file)  # Convert from string if necessary

    # File and path checking
    assert onnx_file.is_file(), 'ONNX file not found: {}'.format(onnx_file)

    plan_file = onnx_file.with_suffix('.plan')

    # Remove the output file if it already exists
    if plan_file.is_file():
        plan_file.unlink()

    # Setup TensorRT builder and create network
    builder = trt.Builder(_LOGGER)
    batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags=batch_flag)

    # Parse the ONNX file
    parser = trt.OnnxParser(network, _LOGGER)
    parser.parse_from_file(str(onnx_file))

    # Define DNN parameters for inference
    builder.max_batch_size = max_batch_size
    config = builder.create_builder_config()
    config.max_workspace_size = max_workspace_size
    if fp16_mode:
        config.set_flag(trt.BuilderFlag.FP16)

    # Optimize the network
    optimized_input_dims = (max_batch_size, input_len)
    profile = builder.create_optimization_profile()
    input_name = input_node_name + input_port_name
    # Set the min, optimal, and max dimensions for the input layer.
    profile.set_shape(input_name, (1, input_len), optimized_input_dims,
                      optimized_input_dims)
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)

    # Write output plan file
    assert engine is not None, 'Unable to create TensorRT engine. Check settings'
    with open(plan_file, 'wb') as file:
        file.write(engine.serialize())

    if verbose:
        # Print information to user
        if os.path.isfile(plan_file):
            print('\nONNX File Name  : {}'.format(onnx_file))
            print('ONNX File Size  : {}'.format(os.path.getsize(onnx_file)))
            print('PLAN File Name : {}'.format(plan_file))
            print('PLAN File Size : {}\n'.format(os.path.getsize(plan_file)))
            print('Network Parameters for inference on AIR-T:')
            print('CPLX_SAMPLES_PER_INFER = {}'.format(int(input_len / 2)))
            print('BATCH_SIZE <= {}'.format(max_batch_size))
        else:
            print('Result    : FAILED - plan file not created')
    return plan_file


def uff2plan(uff_file: Union[str, os.PathLike],
             input_node_name: str = 'input/IteratorGetNext',
             input_len: int = 4096,
             fp16_mode: bool = True,
             max_workspace_size: int = 1073741824,
             max_batch_size: int = 128,
             verbose: bool = False):
    """Optimize the provided UFF (TensorFlow 1.x) model using TensorRT and save the result.

    The optimized model will have a ``.plan`` extension and be saved in the same
    folder as the input model.

    :param uff_file:
        Filename of the UFF model to optimize
    :param input_node_name:
        Name of the UFF model's input layer
    :param input_len:
        Length of the UFF model's input layer, determined when the model was created
    :param fp16_mode:
        Try to use reduced precision (float16) layers if performace would improve
    :param max_workspace_size:
        Maximum scratch memory that the TensorRT optimizer may use, defaults to 1GB.
        The default value can be used in most situations and may only need to be
        reduced if using very low-end GPU hardware
    :param max_batch_size:
        The maximum batch size to optimize for. When running inference using the
        optimized model, the chosen batch size must be less than the maximum
        specified here
    :param verbose:
        Print extra information about the optimized model
    """

    uff_file = pathlib.Path(uff_file)  # Convert from string if necessary

    # File and path checking
    assert uff_file.is_file(), 'UFF file not found: {}'.format(uff_file)

    plan_file = uff_file.with_suffix('.plan')

    # Remove the output file if it already exists
    if plan_file.is_file():
        plan_file.unlink()

    # Input parameters specific to the trained model
    input_node_dims = (1, 1, input_len)  # Input dimensions to trained model

    # Make the plan file
    builder = trt.Builder(trt.Logger(trt.Logger.INFO))
    network = builder.create_network()

    parser = trt.UffParser()
    parser.register_input(input_node_name, input_node_dims)
    parser.parse(str(uff_file), network)

    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size
    builder.fp16_mode = fp16_mode

    engine = builder.build_cuda_engine(network)

    with open(plan_file, 'wb') as f:
        f.write(engine.serialize())

    if verbose:
        # Print information to user
        if os.path.isfile(plan_file):
            print('\nUFF File Name  : {}'.format(uff_file))
            print('UFF File Size  : {}'.format(os.path.getsize(uff_file)))
            print('PLAN File Name : {}'.format(plan_file))
            print('PLAN File Size : {}\n'.format(os.path.getsize(plan_file)))
            print('Network Parameters for inference on AIR-T:')
            print('CPLX_SAMPLES_PER_INFER = {}'.format(int(input_len / 2)))
            print('BATCH_SIZE <= {}'.format(max_batch_size))
        else:
            print('Result    : FAILED - plan file not created')
    return plan_file


def plan_bench(plan_file_name: Union[str, os.PathLike],
               cplx_samples: int,
               batch_size: int = 128,
               num_inferences: Optional[int] = 100,
               input_dtype: np.number = np.float32) -> None:
    """
    Benchmarks a model that has been pre-optimized using the TensorRT framework.

    This function uses settings for the CUDA context and memory buffers that are
    optimized for NVIDIA Jetson modules and may not be optimal for desktops.

    .. note::
        When selecting a ``batch_size`` to benchmark, the selected size must be
        less than or equal to the ``max_batch_size`` value that was specified
        when creating the ``.plan`` file. Additionally, to maximise performance,
        power-of-two values for ``batch_size`` are recommended.

    .. note::
        To accurately benchmark the result of TensorRT optimization, this benchmark
        should be run on the same computer that generated the `.plan` file.

    :param plan_file_name: TensorRT optimized model file (``.plan`` format)
    :param cplx_samples:   Input length of the neural network, in complex samples;
                           this is half of the ``input_length`` of the neural
                           network, which operates on real values
    :param batch_size:     How many sets of ``cplx_samples`` inputs are batched
                           together in a single inference call
    :param num_inferences: Number of iterations to execute inference between
                           measurements of inference throughput
                           (if None, then run forever)
    :param input_dtype:    Data type of a single value (a single I or Q value,
                           not a complete complex (I, Q) sample): use one of
                           :class:`numpy.int16` or :class:`numpy.float32` here
    """

    # Setup the CUDA context
    trt_utils.make_cuda_context()

    # Use PyCUDA to create a shared memory buffer that will receive samples from the
    # AIR-T to be fed into the neural network.
    buff_len = 2 * cplx_samples * batch_size
    sample_buffer = trt_utils.MappedBuffer(buff_len, input_dtype)

    # Set up the inference engine. Note that the output buffers are created for
    # us when we create the inference object.
    dnn = TrtInferFromPlan(plan_file_name, batch_size,
                           sample_buffer)

    # Populate input buffer with test data
    dnn.input_buff.host[:] = np.random.randn(buff_len).astype(input_dtype)

    # Time the DNN Execution
    start_time = time.monotonic()
    for _ in range(num_inferences):
        dnn.feed_forward()
    elapsed_time = time.monotonic() - start_time
    total_cplx_samples = cplx_samples * batch_size * num_inferences

    throughput_msps = total_cplx_samples / elapsed_time / 1e6
    rate_gbps = throughput_msps * 2 * sample_buffer.host.itemsize * 8 / 1e3
    print('Result:')
    print('  Samples Processed : {:,}'.format(total_cplx_samples))
    print('  Processing Time   : {:0.3f} msec'.format(elapsed_time / 1e-3))
    print('  Throughput        : {:0.3f} MSPS'.format(throughput_msps))
    print('  Data Rate         : {:0.3f} Gbit / sec'.format(rate_gbps))


class TrtInferFromPlan:
    """Wrapper class for TensorRT inference using a pre-optimized ``.plan`` file.

    Since it is expensive to create these inference objects, they should be
    created once at the start of your program and then re-used for multiple
    inference calls.

    The buffer containing data for inference is provided when creating this
    inference object and will be re-used for each inference. It's designed to
    be used by repeatedly copying data from the radio into this buffer and then
    calling the :meth:`feed_forward` method to run inference.

    After calling :meth:`feed_forward`, the inference results will be available
    as the :class:`MappedBuffer` object :attr:`output_buff`.

    .. note::
        Only device-mapped memory buffers are supported.
    """

    def __init__(self,
                 plan_file: Union[str, os.PathLike],
                 batch_size: int,
                 input_buffer: MappedBuffer,
                 verbose: bool = True) -> None:
        """
        :param `path_like` plan_file:
            TensorRT ``.plan`` file containing the optimized model
        :param int batch_size:
            Batch size for a single inference execution
        :param MappedBuffer input_buffer:
            Buffer containing data for inference of size ``input_length x batch_size``,
            where ``input_length`` is the length of input to the model,
            determined when the neural network was created
        :param bool verbose:
            Print verbose information about the loaded network, defaults to True

        :var MappedBuffer input_buff:   Input buffer for use in inference
        :var MappedBuffer output_buff:  Output buffer containing inference results
        """

        # Create TensorRT Engine from plan file
        logger_settings = trt.Logger(trt.Logger.VERBOSE)
        deserializer = trt.Runtime(logger_settings).deserialize_cuda_engine
        with open(plan_file, 'rb') as f:
            trt_engine = deserializer(f.read())

        # Perform data size/shape checks
        assert batch_size <= trt_engine.max_batch_size, 'Invalid batch size'
        if batch_size != trt_engine.max_batch_size:
            warnings.warn('Unoptimized batch size detected', RuntimeWarning)
        self._batch_size = batch_size

        # Make assumptions about the input and output binding indexes. These should
        # hold true if your model has one input and one output layer.
        input_binding_index = 0
        input_layer = trt_engine[input_binding_index]
        output_binding_index = 1
        output_layer = trt_engine[output_binding_index]

        # By default, the input and output shape do not account for the batch size.
        # For sanity, we take care of this now, and set the "N" dimension to the
        # batch size. Note that there is also a special case for the input, where
        # if the current N dimension is -1, we will have to later set the binding
        # shape when we create the inference context.
        batch_dim_index = 0
        input_shape = trt_engine.get_binding_shape(input_layer)
        self._explicit_batch = (input_shape[batch_dim_index] == -1)
        input_shape[batch_dim_index] = self._batch_size
        output_shape = trt_engine.get_binding_shape(output_layer)
        output_shape[batch_dim_index] = self._batch_size

        # Now we can sanity check the input buffer provided by the caller. The
        # size of the input buffer should match the expected size from the
        # PLAN file.
        sdr_out_size = input_buffer.host.size
        trt_in_size = trt.volume(input_shape)
        input_dtype = trt.nptype(trt_engine.get_binding_dtype(input_layer))
        assert trt_in_size == sdr_out_size, 'Plan expected {} but got {} ' \
                                            'samples'.format(trt_in_size, sdr_out_size)
        self.input_buff = input_buffer

        # For the output layer, we have a specified size and type from the PLAN
        # file. We use this info to create the output buffer for inference results.
        trt_out_size = trt.volume(output_shape)
        output_dtype = trt.nptype(trt_engine.get_binding_dtype(output_layer))
        self.output_buff = MappedBuffer(trt_out_size, output_dtype)

        # Create inference context
        self._infer_context = trt_engine.create_execution_context()
        # If the PLAN file was created using a network with an explicit batch
        # size (likely as a result of using the ONNX workflow), we have to set
        # the binding shape now so that the batch size is accounted for in the
        # execution context.
        if self._explicit_batch:
            self._infer_context.set_binding_shape(input_binding_index, input_shape)

        if verbose:
            print('TensorRT Inference Settings:')
            print('  Batch Size           : {}'.format(self._batch_size))
            print('  Explicit Batch       : {}'.format(self._explicit_batch))
            print('  Input Layer')
            print('    Name               : {}'.format(input_layer))
            print('    Shape              : {}'.format(input_shape))
            print('    dtype              : {}'.format(input_dtype.__name__))
            print('  Output Layer')
            print('    Name               : {}'.format(output_layer))
            print('    Shape              : {}'.format(output_shape))
            print('    dtype              : {}'.format(output_dtype.__name__))
            print('  Receiver Output Size : {:,} samples'.format(sdr_out_size))
            print('  TensorRT Input Size  : {:,} samples'.format(trt_in_size))
            print('  TensorRT Output Size : {:,} samples'.format(trt_out_size))

    def feed_forward(self) -> None:
        """Forward propagates the input buffer through the neural network
        to run inference.

        Call this method each time samples from the radio are read into
        :attr:`input_buff`. Results will be available afterwards in
        :attr:`output_buff`.
        """
        buffers = [self.input_buff.device, self.output_buff.device]
        # Based on how the PLAN file was generated, we either have already accounted
        # for the batch size or need to specify it again.
        if self._explicit_batch:  # batch size previously accounted for
            self._infer_context.execute_v2(buffers)
        else:
            self._infer_context.execute(self._batch_size, buffers)