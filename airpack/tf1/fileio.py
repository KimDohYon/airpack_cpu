# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
from typing import List, Tuple, Union

import numpy as np
import tensorflow as tf
import uff
import tf2onnx
from tensorflow.python.tools import optimize_for_inference_lib


def bytes2signal(inset: bytes, dtype: tf.dtypes.DType = tf.int16,
                 scalar: float = 1.0) -> tf.Tensor:
    """ Inputs a set of raw data (bytes), decodes it to dtype. It then normalizes if
    the data type is an integer.

    :param inset:   Inputs a set of raw data (bytes)
    :param dtype:   dtype to convert raw data to
    :param scalar:  Scalar gain value (linear units)
    :return:        A Tensor object storing the decoded bytes.
    """

    if dtype is tf.int16:
        signal = tf.cast(tf.decode_raw(inset, dtype), tf.float32) / dtype.max
    elif dtype is tf.float32:
        signal = tf.decode_raw(inset, dtype)
    else:
        raise AssertionError('Unrecognized data type {}. Use tf.int16 or tf.float32'.format(dtype))
    scaled_signal = signal * scalar
    return scaled_signal


def pars_folder(datapath: Union[str, os.PathLike],
                shuffle: bool = True) -> Tuple[List[str], List[int]]:
    """ Recursively find all files with the ".bin" extension in the data path and
    shuffle the data set if requested.

    .. note ::
        The numeric label is the name of the bottom-most folder in the tree of data:
        e.g., for file ``data/train/x/y/file.bin``, the label is ``y``.

    :param datapath:    Directory that the data resides in
    :param shuffle:     Shuffle the data set if requrested
    :return: tuple: ``(filenames, labels)``

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
        # of data: e.g., for file 'data/train/x/y/file.bin', the label is 'y'.
        labels.append(int(curr_filename.parent.stem))

    if shuffle:
        ndx = np.random.permutation(len(filenames))
        filenames = [filenames[i] for i in ndx]
        labels = [labels[i] for i in ndx]

    return filenames, labels


def _sigpipe(datafiles: list,
             labels: list,
             input_len: int,
             output_len: int,
             n_epoch: int,
             batch_size: int,
             dtype: tf.dtypes.DType,
             nthread: int = 4,
             buffer_size: int = 16,
             interweaved: bool = True,
             scalar: float = 1.0) -> tf.compat.v1.data.Iterator:
    """ Data pipeline optimized for reading signal data (I/Q) and feeding it to a deep
    learning model. You likely want to call sigpipe_initialized which is a wrapper for
    this function that initializes the data iterator.

    :param datafiles:       List of data files
    :param labels:          List of labels for each data file
    :param input_len:       Number of complex samples as the input to the neural network.
    :param output_len:      Number of labels possible (defines output layer length)
    :param n_epoch:
    :param batch_size:      Number of batches to read for each iteration
    :param dtype:           dtype of the data in datafiles
    :param nthread:         Number of CPU threads to use when pipelining data reads
    :param buffer_size:     Buffer size used for shuffling the data
    :param interweaved:     Is data interweaved I/Q or not
    :param scalar:          scalar multiplied by signal for data normalization
    :return:                The tensorflow Dataset interator (uninitialized)
    """
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
        dataset = dataset.repeat(n_epoch)
        iterator = dataset.make_initializable_iterator()
    return iterator


def datareader(sess: tf.compat.v1.Session,
               data_folder: Union[str, os.PathLike],
               input_len: int,
               output_len: int,
               n_epoch: int,
               batch_size: int,
               dtype: tf.dtypes.DType = tf.int16,
               nthread: int = 4,
               buffer_size: int = 16,
               interweaved: bool = True,
               scalar: float = 1.0) -> Tuple:
    """ Data pipeline optimized for reading signal data (I/Q) and feeding it to a deep
    learning model.

    Example usage:

    .. code-block:: python

        # Initialize the TensorFlow session
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)
        batch_x, batch_y, nfiles = fileio.datareader(
            sess,
            data_path,
            input_len = 2048,
            output_len = 12,
            n_epoch = 10,
            batch_size = 128
        )
        # Read the data in a while loop.
        try:
            while True:
                data, labels = sess.run([batch_x, batch_y])
        except tf.errors.OutOfRangeError:  # exception returned when n_epoch exceeded
            pass


    :param sess:            Tensorflow Session
    :param data_folder:     Directory that the data resides in
    :param input_len:       Number of complex samples as the input to the neural network.
    :param output_len:      Number of labels possible (defines output layer length)
    :param n_epoch:         Number of epochs before tf.errors.OutOfRangeError is thrown
    :param batch_size:      Number of batches to read for each iteration
    :param dtype:           dtype of the data in datafiles
    :param nthread:         Number of CPU threads to use when pipelining data reads
    :param buffer_size:     Buffer size used for shuffling the data
    :param interweaved:     Is data interweaved I/Q or not
    :param scalar:          scalar multiplied by signal for data normalization
    :return:                ``(batch_x, batch_y, iterator)`` where
                            batch_x = iterator of training files
                            batch_y = iterator of labels for training files
                            nfiles = total number of files
    """

    # Initialize the data files and labels
    datafiles, labels = pars_folder(data_folder)
    nfiles = len(datafiles)
    with tf.name_scope('input'):
        iterator = _sigpipe(datafiles, labels, input_len, output_len, n_epoch, batch_size,
                            dtype=dtype, nthread=nthread, buffer_size=buffer_size,
                            interweaved=interweaved, scalar=scalar)
        sess.run(iterator.initializer)  # initialize session and iterator
        [batch_x, batch_y] = iterator.get_next()
    return batch_x, batch_y, nfiles


class TfSaver:
    """Class that saves the tensorflow 1 session during training.

        Once created, the :py:meth:`save` field can be used to save a model periodically
        during training and once the model is fully trained.

        Example usage:

        .. code-block:: python

            # Initialize the TensorFlow session
            init = tf.compat.v1.global_variables_initializer()
            sess.run(init)
            saver = fileio.TfSaver(model_save_folder)
            while True:
                # <train model here>
                saver.save(sess)
        """
    def __init__(self, saver_dir: Union[str, os.PathLike]):
        """
        :param saver_dir:     directory to save the TensorFlow model
        """
        self._saver = tf.compat.v1.train.Saver(name='saver')
        self._saver_path = saver_dir
        if not os.path.isdir(self._saver_path):
            os.makedirs(self._saver_path, exist_ok=True)
        self.saver_name = os.path.join(self._saver_path, 'saved_model')

    def save(self, tf_session: tf.compat.v1.Session):
        """ Save state of tensorflow session

        :param tf_session:  TensorFlow Session to save the state of.
        :return:
        """
        self._saver.save(tf_session, self.saver_name)


def sess2uff(sess: tf.compat.v1.Session,
             in_node_name: str = 'input/IteratorGetNext',
             out_node_name: str = 'output/networkout',
             filename: Union[str, os.PathLike] ='saved_model.uff',
             addsoftmax: bool = False,
             quiet: bool = False):
    """ Convert a TensorFlow session to a UFF file.

    This function is used to freeze a TensorFlow 1 model and export it into a format
    that TensorRT can read in to convert to an optimized .plan file for deployment on
    the AIR-T

    .. note::
        The values for :attr:`in_node_name` and :attr:`out_node_name` are defined in the
        TensorFlow 1 model using the :py:func:`tf.name_scope`.

    :param sess:            TensorFlow Session
    :param in_node_name:    Input node name
    :param out_node_name:   Output node name
    :param filename:        Name of file to export
    :param addsoftmax:      Add a :py:func:`tf.nn.softmax` to the output. Some training
                            methods have this incorporated so it must be added after
                            training
    :param quiet:           Don't be verbose
    :return:
    """
    if addsoftmax:
        tf.nn.softmax(sess.graph.get_tensor_by_name(out_node_name + ':0'), name='prediction')
        out_node_name = 'prediction'
    graph_def = sess.graph.as_graph_def()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess,
                                                                    graph_def,
                                                                    [out_node_name])
    opt_graph_def = optimize_for_inference_lib.optimize_for_inference(frozen_graph_def,
                                                                      [in_node_name],
                                                                      [out_node_name],
                                                                      tf.float32.as_datatype_enum)
    uff.from_tensorflow(opt_graph_def, [out_node_name], quiet=quiet, output_filename=filename)


def sess2onnx(sess: tf.compat.v1.Session,
              in_node_name: str = 'input/IteratorGetNext',
              out_node_name: str = 'output/networkout',
              filename: Union[str, os.PathLike] = 'saved_model.onnx',
              addsoftmax: bool = False):
    """ Convert a TensorFlow session to an ONNX file.

    This function is used to freeze a TensorFlow 1 model and export it into a format
    that TensorRT can read in to convert to an optimized .plan file for deployment on
    the AIR-T.

    .. note::
        The values for :attr:`in_node_name` and :attr:`out_node_name` are defined in the
        TensorFlow 1 model using the :py:func:`tf.name_scope`.

    :param sess:            TensorFlow Session
    :param in_node_name:    Input node name
    :param out_node_name:   Output node name
    :param filename:        Name of file to export
    :param addsoftmax:      Add a :py:func:`tf.nn.softmax` to the output. Some training
                            methods have this incorporated so it must be added after
                            training
    :return:
    """
    if addsoftmax:
        tf.nn.softmax(sess.graph.get_tensor_by_name(out_node_name + ':0'),
                      name='prediction')
        out_node_name = 'prediction'
    graph_def = sess.graph.as_graph_def()
    frozen_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        [out_node_name]
    )
    model_proto, _ = tf2onnx.convert.from_graph_def(
        frozen_graph_def,
        name=None,
        input_names=[in_node_name + ':0'],
        output_names=[out_node_name + ':0'],
        opset=12,
        custom_ops=None,
        custom_op_handlers=None,
        custom_rewriter=None,
        inputs_as_nchw=None,
        extra_opset=None,
        shape_override=None,
        target=None,
        large_model=False,
        output_path=None
    )
    with open(filename, 'wb') as f:
        f.write(model_proto.SerializeToString())
    print(filename)
