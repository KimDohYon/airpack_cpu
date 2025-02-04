#!/usr/bin/env python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
from typing import Union

import keras2onnx
import onnx
import tensorflow as tf

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

try:
    from airpack.tf2 import fileio, model
except ModuleNotFoundError as e:
    import sys
    _msg = "{0}\nPlease run:\n  pip install -e {1}".format(e, _airpack_root)
    raise ModuleNotFoundError(_msg).with_traceback(sys.exc_info()[2]) from None


for dev in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(dev, True)


def train(data_folder: Union[str, os.PathLike], n_epoch: int = 10) -> float:
    """ Script used to train the :py:func:`airpack.tf2.model.default_network`.

    .. note ::
        You may modify the parameters in this script to tune the hyperparameters of the
        :py:func:`airpack.tf2.model.default_network`.

    :param data_folder:     Location of training data
    :param n_epoch:         Number of epochs in training process
    :return:                Training accuracy
    """

    # Define Model Parameters
    data_folder = pathlib.Path(data_folder)
    train_data_folder = data_folder / "train"   # Location of training data
    test_data_folder = data_folder / "test"    # Location of training data (validation subset)
    model_save_folder = _airpack_root / "output" / "tf2"
    os.makedirs(model_save_folder, exist_ok=True)

    input_len = 2048  # signal Window length in complex samples
    output_len = 12  # Number of signal classes
    learning_rate = 1e-3    # Learning rate
    dtype = tf.int16

    # Define Training Parameters
    batch_size = 128  # Number of images to average for each training calculation
    normalize_scalar = 1000  # Scalar multiplied by input to normalize the data

    # Import Training Data Set
    train_data = fileio.datareader(train_data_folder, input_len, output_len,
                                   batch_size, dtype=dtype, nthread=4,
                                   buffer_size=16, interweaved=True,
                                   scalar=normalize_scalar)
    # Import Test Data Set
    test_data = fileio.datareader(test_data_folder, input_len, output_len,
                                  batch_size, dtype=dtype, nthread=4,
                                  buffer_size=16, interweaved=True,
                                  scalar=normalize_scalar)

    # Define Feed forward CNN model
    classifier = model.default_network(input_len, output_len)

    # Define Training Optimization Function
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    metrics = [tf.keras.metrics.CategoricalAccuracy()]

    classifier.compile(optimizer=optimizer,
                       loss=tf.nn.softmax_cross_entropy_with_logits,
                       metrics=metrics)

    # Train model
    history = classifier.fit(x=train_data, epochs=n_epoch,
                             validation_data=test_data, verbose=1)

    # Save model (could be called during training to incrementally save the model)
    onnx_model = keras2onnx.convert_keras(classifier, classifier.name, target_opset=13)
    model_file = model_save_folder / "saved_model.onnx"
    onnx.save_model(onnx_model, model_file)
    accuracy = history.history['categorical_accuracy'][-1]
    return accuracy


if __name__ == '__main__':
    _default_data_folder = '/data'
    train(_default_data_folder)
