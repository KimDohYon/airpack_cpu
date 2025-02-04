#!/usr/bin/env python3

# Copyright (C) 2021 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

"""
Tools for deploying trained neural networks and benchmarking their performance.

In this sense, "deploying" a neural network model means running it using a standalone inference
engine (and not from within the learning framework that was used to define and train the model).

This module supports two different inference engines: `onnxruntime` and `TensorRT`. For deployment
on an AIR-T, it is currently recommended to use `TensorRT` for maximum performance, but `onnxruntime`
is quite good for quickly testing a model during development or testing a model on the server used
to train it.
"""
