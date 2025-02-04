#!/usr/bin/env python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL
# SOFTWARE SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the
# license was not received, please write to support@deepwavedigital.com

"""
Tools for deploying trained neural networks on the AIR-T.

In this sense, "deploying" a neural network model means running it using a standalone
inference engine (and not from within the learning framework that was used to define and
train the model).

The method provided here leverages the PyCUDA interface for the shared memory buffer.
PyCUDA is installed by default in AirStack, however Deepwave recommends using an
Anaconda environment based on the ``environments/airpack_airt.yml`` file provide with
AirPack.
"""
