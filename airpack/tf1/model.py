# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com
#
# For help on TensorFlow Functions, see the API: https://www.tensorflow.org/api_docs/python/

"""
Contains neural network models for signal classification implemented in the Tensorflow
1.x framework (using :py:mod:`tf.nn` objects).

When getting started with this package, you can use these models as-is: the default
sizes and parameter values are reasonable for the bundled datasets. Next, you can
explore how tuning the values provided as arguments to these models ("model
hyperparameters") affects their performance and accuracy. Finally, these models are each
standalone functions, so you can copy them into your own code to add layers, change
activation functions, and experiment with different model structures entirely!

"""

from typing import Optional
import tensorflow as tf
import numpy as np


def default_network(x: tf.Tensor,
                    input_len: int,
                    output_len: int,
                    num_filt: Optional[int] = None,
                    filt_size: Optional[list] = None,
                    pool_size: Optional[list] = None,
                    fc_nodes: Optional[list] = None):
    """ Creates a convolutional signal classifier model for chunks of signal
    data that are a specified number of samples long.

    Required arguments are the input tensor, input length (number of samples in
    the input), and the number of classes of signal in the classifier output.
    Other optional arguments are tunable hyperparameters to optimize the model
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

        Please see the TensorFlow 1.x documentation for the
        :class:`tf.compat.v1.nn.conv2d` class for more information on
        2-D convolutional layers in neural networks.

    :param x:            Input tensor of data
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
    if filt_size is None:
        filt_size = [2, 32]
    if pool_size is None:
        pool_size = [1, 16]
    if fc_nodes is None:
        fc_nodes = [128, 64, 32]
    assert len(filt_size) == 2, 'filter size must have length of 2'
    assert len(pool_size) == 2, 'pool size must have length of 2'
    assert len(fc_nodes) == 3, 'Please specify only 3 nodes for the fully connected layers'

    with tf.name_scope('reshape'):
        # Input data is interweaved I/Q. Reshape into a matrix of shape =
        # [batch_size, 2, input_len, 1]. It is not advised to modify this layer.
        x_reshape = tf.transpose(tf.reshape(x, [-1, input_len, 2, 1]), perm=[0, 2, 1, 3])

    with tf.name_scope('conv0'):
        # First convolution layer followed by max pooling. Additional convolution layers
        # may be added by copying this one.
        # Layer construction:
        w0_shape = [filt_size[0], filt_size[1], 1, num_filt]
        b0_shape = [num_filt]
        w0 = tf.Variable(tf.random.truncated_normal(w0_shape, stddev=0.1, name='weights'))
        b0 = tf.Variable(tf.constant(0.0, shape=b0_shape, name='biases'))
        conv0 = _conv2d_relu(x_reshape, w0, b0)
        pool0 = _maxpool(conv0, k=pool_size)
        pool0_flat = tf.reshape(pool0, [-1, np.prod(pool0.shape.as_list()[-3:])])

    with tf.name_scope('fc1'):
        # The first dimension of the input shape of this layer may be defined by the output shape
        # of the previous layer. The second dimension is the number of nodes
        # Layer construction.
        # The shape of the bias tensor is defined by the output shape of the previous layer
        w1_shape = [pool0_flat.shape.as_list()[1], fc_nodes[0]]
        b1_shape = [fc_nodes[0]]
        w1 = tf.Variable(tf.random.truncated_normal(w1_shape, stddev=0.1, name='weights'))
        b1 = tf.Variable(tf.constant(0.0, shape=b1_shape, name='biases'))
        fc1 = _fc_relu(pool0_flat, w1, b1)

    with tf.name_scope('fc2'):
        # The first dimension of the input shape of this layer may be defined by the output shape
        # of the previous layer. The second dimension is the number of nodes
        # Layer construction.
        # The shape of the bias tensor is defined by the output shape of the previous layer
        w2_shape = [fc1.shape.as_list()[1], fc_nodes[1]]
        b2_shape = [fc_nodes[1]]
        w2 = tf.Variable(tf.random.truncated_normal(w2_shape, stddev=0.1, name='weights'))
        b2 = tf.Variable(tf.constant(0.0, shape=b2_shape, name='biases'))
        fc2 = _fc_relu(fc1, w2, b2)

    with tf.name_scope('fc3'):
        # The first dimension of the input shape of this layer may be defined by the output shape
        # of the previous layer. The second dimension is the number of nodes
        # Layer construction.
        # The shape of the bias tensor is defined by the output shape of the previous layer
        w3_shape = [fc2.shape.as_list()[1], fc_nodes[2]]
        b3_shape = [fc_nodes[2]]
        w3 = tf.Variable(tf.random.truncated_normal(w3_shape, stddev=0.1, name='weights'))
        b3 = tf.Variable(tf.constant(0.0, shape=b3_shape, name='biases'))
        fc3 = _fc_relu(fc2, w3, b3)

    with tf.name_scope('output'):
        # The first dimension of the input shape of this layer may be defined by the output shape
        # of the previous layer. The second dimension is the number of nodes
        w4_shape = [fc3.shape.as_list()[1], output_len]
        b4_shape = [output_len]
        w4 = tf.Variable(tf.random.truncated_normal(w4_shape, stddev=0.1, name='weights'))
        b4 = tf.Variable(tf.constant(0.0, shape=b4_shape, name='biases'))
        # There is no activation function here because the optimizer algorithm in run_training.py
        # includes a softmax activation function. You will have to add the softmax layer back in
        # for deployment since the optimizer will not be used at that point. It is only used for
        # training.
        prediction = _fc(fc3, w4, b4, name='networkout')

    return prediction


# Variable Initialization Functions
def _conv2d_relu(x: tf.Tensor,
                 w: tf.Variable,
                 b: tf.Variable,
                 strides: int = 1) -> tf.Tensor:
    """ Two dimensional convolution layer with relu activation

    :param x:           Input tensor
    :param w:           weights for each node of the layer
    :param b:           biases for each node of the layer
    :param strides:     strides to use for the convolution
    :return:            Output tensor
    """
    x1 = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='VALID')
    x2 = tf.nn.bias_add(x1, b)
    return tf.nn.relu(x2)


def _maxpool(x: tf.Tensor,
             k: tuple = (1, 2)) -> tf.Tensor:
    """ Max pooling layer

    :param x:   Input tensor
    :param k:   Pool size
    :return:    Output tensor
    """
    return tf.nn.max_pool2d(x, ksize=[1, k[0], k[1], 1], strides=[1, k[0], k[1], 1],
                            padding='VALID')


def _fc_relu(x: tf.Tensor,
             w: tf.Variable,
             b: tf.Variable,
             name: str = 'fc_relu') -> tf.Tensor:
    """ Fully connected layer function with RELU activation

    :param x:       Input tensor
    :param w:       Weights for each node of the layer
    :param b:       Biases for each node of the layer
    :param name:    Name of the relu (final) layer
    :return:        Output tensor
    """
    return tf.nn.relu(tf.add(tf.matmul(x, w), b), name=name)


def _fc(x: tf.Tensor,
        w: tf.Variable,
        b: tf.Variable,
        name: str = 'fc'):
    """ Fully connected layer set (multiply, add) with no activation.

    :param x:       Input tensor
    :param w:       Weights for each node of the layer
    :param b:       Biases for each node of the layer
    :param name:    Name of the add (final) layer
    :return:        Output tensor
    """
    return tf.add(tf.matmul(x, w), b, name=name)
