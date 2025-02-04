#!/usr/bin/env python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

"""Executable scripts for Tensorflow 1.x backend for AirPack.

When this module is imported, it will check to make sure that the version
of TensorFlow present on your system is compatible with this backend.
"""

import pkg_resources
import tensorflow

_parse_version = pkg_resources.packaging.version.parse
_tf_version = _parse_version(tensorflow.__version__)
if _tf_version < _parse_version('1.14.0'):
    raise ImportError(
        "airpack_scripts.tf1 requires at least Tensorflow 1.14.0, found version {}".format(
            tensorflow.__version__))
elif _tf_version >= _parse_version('2.0.0'):
    raise ImportError(
        "airpack_scripts.tf1 requires Tensorflow 1.x, but found version {}".format(
            tensorflow.__version__))
