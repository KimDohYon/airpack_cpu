# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib


def bytes2signal(inset: bytes, dtype: tf.dtypes.DType = tf.int16,
                 scalar: float = 1.0) -> tf.Tensor:
    """ Inputs a set of raw data (bytes), decodes it to :attr:`dtype`. It then normalizes
    to be between ``(-1, 1)`` if the data type is an integer.

    :param inset:   Inputs a set of raw data (bytes)
    :param dtype:   dtype to convert raw data to (tf.int16 or tf.float32)
    :param scalar:  Scalar gain value (linear units)
    :return:        A Tensor object storing the decoded bytes.
    """

    if dtype is tf.int16:
        signal = tf.cast(tf.io.decode_raw(inset, dtype), tf.float32) / dtype.max
    elif dtype is tf.float32:
        signal = tf.io.decode_raw(inset, dtype)
    else:
        raise AssertionError(f'Unrecognized dtype {dtype}. Use tf.int16 or tf.float32')
    scaled_signal = signal * scalar
    return scaled_signal


def pars_folder(datapath: Union[str, os.PathLike],
                shuffle: bool = True) -> Tuple[List[str], List[int]]:
    """ Recursively find all files with the ".bin" extension in the data path and
    shuffle the data set if requested.

    .. note ::
        The numeric label is the name of the bottom-most folder in the tree of data:
        e.g., for file 'data/train/x/y/file.bin', the label is 'y'.

    :param datapath:    Directory that the data resides in
    :param shuffle:     Shuffle the data set if requrested
    :return:            ``(filenames, labels)``

    """

    # Go through the data_folder and find the data files
    datapath = pathlib.Path(datapath)
    assert datapath.is_dir(), 'datapath does not exist: %s' % datapath

    filenames = list()
    labels = list()

    # Recursively find all files with the ".bin" extension in the data path.
    for f in datapath.rglob("*.bin"):
        curr_filename = f.absolute()
        filenames.append(str(curr_filename))
        # The numeric label is the name of the bottom-most folder in the tree
        # of data: e.g., for file ``data/train/x/y/file.bin``, the label is ``y``.
        labels.append(int(curr_filename.parent.stem))

    if shuffle:
        ndx = np.random.permutation(len(filenames))
        filenames = [filenames[i] for i in ndx]
        labels = [labels[i] for i in ndx]

    return filenames, labels


def datareader(data_folder: Union[str, os.PathLike], input_len: int, output_len: int,
               batch_size: int, dtype=tf.dtypes.DType, nthread: int = 4,
               buffer_size: int = 16, interweaved: bool = True,
               scalar: float = 1.0) -> tf.data.Dataset:
    """ Data pipeline optimized for reading signal data (I/Q) and feeding it to a deep
    learning model.

    :param data_folder:     Directory that the data resides in
    :param input_len:       Number of complex samples as the input to the neural network.
    :param output_len:      Number of labels possible (defines output layer length)
    :param batch_size:      Number of batches to read for each iteration
    :param dtype:           dtype of the data in datafiles
    :param nthread:         Number of CPU threads to use when pipelining data reads
    :param buffer_size:     Buffer size used for shuffling the data
    :param interweaved:     Is data interweaved I/Q or not
    :param scalar:          scalar multiplied by signal for data normalization
    :return:                The tensorflow Dataset object
    """

    # Initialize the data files and labels
    datafiles, labels = pars_folder(data_folder)
    assert len(datafiles) > 0, 'list of datafiles is empty.'
    sample_bytes = dtype.size
    if interweaved:
        record_bytes = 2 * input_len * sample_bytes
    else:
        record_bytes = input_len * sample_bytes
    with tf.name_scope('InputDataSet'):  # Create input dataset pipeline
        inset = tf.data.FixedLengthRecordDataset(datafiles, record_bytes)
        inset = inset.map(lambda x: bytes2signal(x, dtype=dtype, scalar=scalar),
                          num_parallel_calls=nthread)
    with tf.name_scope('OutputLabels'):  # Create label dataset pipeline
        outset = tf.data.Dataset.from_tensor_slices(labels)
        outset = outset.map(lambda x: tf.one_hot(x, output_len))
    with tf.name_scope('Iterator'):  # Merge inset and outset into a single pipeline call
        dataset = tf.data.Dataset.zip((inset, outset))
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size)
    with tf.name_scope('input'):
        return dataset
