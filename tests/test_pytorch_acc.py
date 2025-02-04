#!/usr/bin/env python3
# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import pathlib
import sys

import numpy as np
import pytest

if __name__ == "__main__":
    rc = pytest.main([__file__, "-ra"])
    sys.exit(rc)

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent

try:
    from airpack_scripts.pytorch import run_training, run_inference, make_plan_file
except ImportError:
    pytest.skip("PyTorch not available, skipping related tests",
                allow_module_level=True)

FRAMEWORK = 'Pytorch'
_data_folder = '/data'
_onnx_file = _airpack_root / "output" / "pytorch" / "saved_model.onnx"


def test_train_accuracy(accuracy_min=0.8):
    accuracy = run_training.main(_data_folder, n_epoch=2)
    assert accuracy > accuracy_min, \
        f'{FRAMEWORK} training test FAILED: Accuracy = {accuracy} < {accuracy_min}'
    print(f'Passed {FRAMEWORK} Training Test: Accuracy = {accuracy} > {accuracy_min}')


def test_infer_accuracy(min_num_correct=7):
    result = run_inference.infer(_data_folder)
    truth = np.arange(12)
    num_correct = np.sum((np.array(result) == truth).astype(int))
    assert num_correct > min_num_correct, \
        f'{FRAMEWORK} inference test FAILED: Correct Classes = {num_correct} < {min_num_correct}'
    print(f'Passed {FRAMEWORK} Inference Test: Correct Classes = {num_correct} > {min_num_correct}')


def test_make_plan_file():
    plan_file = pathlib.Path(make_plan_file.convert(_onnx_file))
    assert plan_file.exists(), f'{FRAMEWORK} Make Plan File FAILED'
    print(f'Passed {FRAMEWORK} Make Plan File')
