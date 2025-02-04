# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com or:
#
# For help on TensorFlow Functions, see the API: https://www.tensorflow.org/api_docs/python/

"""
Contains neural network models for signal classification implemented
in the Tensorflow 2.x framework (using :py:mod:`tf.layers.keras` objects).

When getting started with this package, you can use these models as-is:
the default sizes and parameter values are reasonable for the bundled
datasets. Next, you can explore how tuning the values provided as
arguments to these models ("model hyperparameters") affects their
performance and accuracy. Finally, these models are each standalone
functions, so you can copy them into your own code to add layers, change
activation functions, and experiment with different model structures
entirely!

"""

from typing import Optional
import tensorflow as tf


def default_network(input_len: int,
                    num_classes: int,
                    num_filt: Optional[int] = None,
                    strides: Optional[int] = None,
                    filt_size: Optional[list] = None,
                    pool_size: Optional[list] = None,
                    fc_nodes: Optional[list] = None) -> tf.keras.Model:
    """ Creates a convolutional signal classifier model for chunks of signal
    data that are a specified number of samples long.

    Required arguments are the input length (number of samples in the input)
    and the number of classes of signal in the classifier output. Other
    optional arguments are tunable hyperparameters to optimize the model
    for specific datasets and/or types of signals. When the optional arguments
    are set to ``None``, default values will be used that will work reasonably
    well.

    This model object is created with default-initialized weights and must be
    first trained before being used for inference.

    .. note ::
        Tensorflow models deal with complex input data by taking as input
        arrays of interleaved I/Q values, then reshaping them into a I/Q
        channel dimension and a sample number dimension for processing.
        This can lead to some confusion about what numbers are used for
        input lengths.

        Here's the relationships you will need to know:

        1. Specify :attr:`input_len` as the number of complex samples to process.
        2. The actual input buffer passed to training/inference will be
           real-valued and of size ``2 * input_len``.
        3. The input buffer is reshaped to a 2-D array of size ``[2, input_len]``
           when passed through the neural network.

    .. seealso ::
        The :attr:`num_filt`, :attr:`strides`, :attr:`filt_size`,
        and :attr:`pool_size` arguments are tunable hyperparameters
        that set the shape of convolutional layers within the model.

        When specifying these sizes, note that these layers operate on
        2-D inputs of size ``[2, N]``, where the first dimension of length
        two refers to the I vs. Q channel, and the second dimension is the
        number of samples :attr:`input_len`.

        Please see the TensorFlow 2.x documentation for the
        :class:`tf.keras.layers.Conv2D` class for more information on
        2-D convolutional layers in neural networks.

    :param input_len:    Number of complex samples to be input to the model
    :param num_classes:  Number of output classes for this classifier
    :param num_filt:     Number of output filters in the convolution
    :param strides:      Stride within convolutional layers
    :param filt_size:    Kernel/window size of convolutional layers,
                         typically ``[2, X]`` for some ``X``
    :param pool_size:    Kernel/window size of downsampling layers,
                         typically ``[1, X]`` for some ``X``
    :param fc_nodes:     List of sizes of fully-connected (FC) layers

    :returns: Signal classifier model object
    """

    # Default values for tunable hyperparameters:
    if num_filt is None:
        num_filt = 16
    if strides is None:
        strides = 1
    if filt_size is None:
        filt_size = [2, 32]
    if pool_size is None:
        pool_size = [1, 16]
    if fc_nodes is None:
        fc_nodes = [128, 64, 32]
    assert len(filt_size) == 2, 'filter size must have length of 2'
    assert len(pool_size) == 2, 'pool size must have length of 2'
    trunc_normal = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
    zero_init = tf.keras.initializers.Zeros()

    m = tf.keras.Sequential()

    # Input is complex data, so 2 * input_len, interleaved I/Q
    input_shape = (2*input_len,)

    m.add(tf.keras.layers.InputLayer(name="input", input_shape=input_shape))

    # Reshape data to Nx2, then permute to de-interleave I/Q samples:
    # input: IQIQIQ output: III in one column QQQ in other column
    m.add(tf.keras.layers.Reshape(name="InputIQ/reshape",
                                  target_shape=(input_len, 2, 1)))
    m.add(tf.keras.layers.Permute(name="InputIQ/deinterleave", dims=(2, 1, 3)))

    # First convolutional layer followed by max pooling and flatten
    m.add(tf.keras.layers.Conv2D(name="Convolution/conv2d",
                                 filters=num_filt,
                                 kernel_size=filt_size,
                                 activation="relu",
                                 strides=strides,
                                 kernel_initializer=trunc_normal,
                                 bias_initializer=zero_init,
                                 use_bias=True,
                                 padding="valid"))
    m.add(tf.keras.layers.MaxPool2D(name="Convolution/maxpool",
                                    pool_size=pool_size,
                                    padding="valid"))
    m.add(tf.keras.layers.Flatten(name="Convolution/flatten"))

    # Start of fully connected layers
    # Size for each layer, len(fc_num_nodes) = # of fully connected layers
    for index, num_nodes in enumerate(fc_nodes):
        m.add(tf.keras.layers.Dense(name="FC{}".format(index),
                                    units=num_nodes,
                                    activation="relu",
                                    kernel_initializer=trunc_normal,
                                    bias_initializer=zero_init,
                                    use_bias=True))

    # Output layer
    m.add(tf.keras.layers.Dense(name="output",
                                units=num_classes,
                                kernel_initializer=trunc_normal,
                                bias_initializer=zero_init,
                                use_bias=True))

    # Return completed model
    return m
