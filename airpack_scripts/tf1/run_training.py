#! /usr/bin/python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
from typing import Union

import tensorflow as tf
import numpy as np

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

try:
    from airpack.tf1 import fileio, model
except ModuleNotFoundError as e:
    import sys
    _msg = "{0}\nPlease run:\n  pip install -e {1}".format(e, _airpack_root)
    raise ModuleNotFoundError(_msg).with_traceback(sys.exc_info()[2]) from None


def train(data_folder: Union[str, os.PathLike], n_epoch: int = 10) -> float:
    """ Script used to train the :py:func:`airpack.tf1.model.default_network`.

    .. note ::
        You may modify the parameters in this script to tune the hyperparameters of the
        :py:func:`airpack.tf1.model.default_network`.

    :param data_folder:     Location of training data
    :param n_epoch:         Number of epochs in training process
    :return:                Training accuracy
    """

    # Define Model Parameters
    data_folder = pathlib.Path(data_folder)
    train_data_folder = data_folder / "train"   # Location of training data
    test_data_folder = data_folder / "test"    # Location of training data (validation subset)
    model_save_folder = _airpack_root / "output" / "tf1"

    input_len = 2048  # signal Window length in complex samples
    output_len = 12  # Number of signal classes
    learning_rate = 1e-3    # Learning rate
    dtype = tf.int16

    # Define Training Parameters
    batch_size = 128  # Number of images to average for each training calculation
    print_period = 100  # How often to print to screen
    normalize_scalar = 1000  # Scalar multiplied by input to normalize the data

    # Create session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.compat.v1.get_default_graph()
    sess = tf.compat.v1.Session(config=config)

    # Import Training Data Set
    train_x, train_y, train_n = fileio.datareader(sess, train_data_folder, input_len, output_len,
                                                  n_epoch, batch_size, dtype=dtype, nthread=4,
                                                  buffer_size=16, interweaved=True,
                                                  scalar=normalize_scalar)
    # Import Test Data Set
    test_x, test_y, test_n, = fileio.datareader(sess, test_data_folder, input_len, output_len,
                                                n_epoch, batch_size, dtype=dtype, nthread=4,
                                                buffer_size=16, interweaved=True,
                                                scalar=normalize_scalar)

    # Define Feed forward CNN model
    prediction = model.default_network(train_x, input_len, output_len)

    # Define Loss Function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction,
                                                                     labels=train_y), name='loss')

    # Define Training Optimization Function
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name='optimizer')

    # Initialize the TensorFlow session
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)

    input_tensor_node = graph.get_tensor_by_name('input/IteratorGetNext:0')

    # Train Model
    n_iter = int(n_epoch * train_n / batch_size) + 1
    step = 0
    try:
        while True:
            # Perform back propagation, ie., train model
            a, l = sess.run([optimizer, loss])
            if step % print_period == 0:  # Test accuracy with test data set
                batch_x, batch_y = sess.run([test_x, test_y])
                test_prediction = sess.run(prediction, feed_dict={input_tensor_node: batch_x})
                result_vals = np.argmax(test_prediction, axis=1)
                truth_vals = np.argmax(batch_y, axis=1)
                accuracy = (result_vals == truth_vals).mean()
                print('(%d of %d): Training Loss = %f, Testing Accuracy = %f' % (step, n_iter, l, accuracy))
            step += 1
    except (tf.errors.OutOfRangeError, KeyboardInterrupt):
        batch_x, batch_y = sess.run([test_x, test_y])
        test_prediction = sess.run(prediction, feed_dict={input_tensor_node: batch_x})
        result_vals = np.argmax(test_prediction, axis=1)
        truth_vals = np.argmax(batch_y, axis=1)
        accuracy = (result_vals == truth_vals).mean()
        print('(%d of %d): Training Loss = %f, Testing Accuracy = %f' % (step, n_iter, l, accuracy))
        pass

    # Save model (could be called during training to incrementally save the model)
    # Define model saver class (Note that this could be called during training for more save points)
    saver = fileio.TfSaver(model_save_folder)
    saver.save(sess)

    # Output model to ONNX file format for deployment on AIR-T
    fileio.sess2onnx(sess, filename=(saver.saver_name + '.onnx'))
    return accuracy


if __name__ == '__main__':
    _default_data_folder = '/data'
    train(_default_data_folder)
