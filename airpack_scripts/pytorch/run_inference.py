#!/usr/bin/env python3

# Copyright (C) 2022 Deepwave Digital, Inc - All Rights Reserved
# You may use, distribute and modify this code under the terms of the DEEPWAVE DIGITAL SOFTWARE
# SOURCE CODE TERMS OF USE, which is provided with the code. If a copy of the license was not
# received, please write to support@deepwavedigital.com

import os
import pathlib
import random
from typing import Any, Callable, Dict, List, Union

from matplotlib import pyplot as plt
import numpy as np
import onnxruntime
import scipy.special

_script_dir = pathlib.Path(__file__).parent.absolute()
_airpack_root = _script_dir.parent.parent

def get_file_pars(filename: Union[str, os.PathLike]) -> Dict[str, Any]:
    file_base = os.path.splitext(os.path.basename(filename))[0]
    file_pars = {}
    for str in file_base.split('_'):
        key, val = str.split('=')
        file_pars[key] = int(val) if val.lstrip('-').isnumeric() else val
    return file_pars

def setup_inference_function(saver_path: pathlib.Path,
                             file_name: str = 'saved_model.onnx') \
                            -> Callable[[np.ndarray], List[float]]:
    onnx_file = saver_path / file_name
    sess = onnxruntime.InferenceSession(str(onnx_file))
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name

    def infer_func(x):
        return sess.run([label_name], {input_name: x})[0]
    return infer_func

def infer(data_folder: Union[str, os.PathLike],
          plot_snr: int = 12,
          fs: float = 122.88e6) -> List[float]:
    test_data_folder = pathlib.Path(data_folder) / "test"
    model_save_folder = _airpack_root / "output" / "pytorch"

    # 데이터 타입과 정규화 스케일러 설정
    dtype = np.float32
    normalize_scalar = 1000

    # 모델 복원
    perform_inference = setup_inference_function(model_save_folder)

    labels = os.listdir(test_data_folder)
    labels.sort()
    files = []
    for label in labels:
        label_folder = test_data_folder / label
        label_files = os.listdir(label_folder)
        random.shuffle(label_files)
        for label_file in label_files:
            pars = get_file_pars(label_file)
            if pars['snr'] == plot_snr:
                files.append(test_data_folder / label / label_file)
                break

    sigs = []
    results = []
    for file in files:
        # float32 데이터 읽기
        sig_interleaved = np.fromfile(file, dtype=dtype)

        # 스케일링 적용
        sig_scaled = sig_interleaved * normalize_scalar

        # 복소수 쌍 생성 (38400 쌍)
        sig = sig_scaled[:38400 * 2:2] + 1j * sig_scaled[1:38400 * 2:2]
        
        # 모델 입력 데이터 준비
        sig_input = np.expand_dims(sig_scaled[:38400 * 2], 0)

        # 추론 수행
        result_array = perform_inference(sig_input)

        # 소프트맥스를 통한 확률 계산
        result_prob = scipy.special.softmax(result_array)
        result = np.argmax(result_prob)

        # 결과 저장
        sigs.append(sig)
        results.append(result)

    # 시각화 및 결과 저장
    plt.style.use('dark_background')
    fig, axs = plt.subplots(3, len(results), figsize=(20, 10),
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.25})
    axs = np.rollaxis(axs, 1)

    axs[0, 0].set_ylabel('Amp (Norm)')
    axs[0, 1].set_ylabel('Freq (MHz)')
    axs[0, 2].set_ylabel('PSD (dB)')
    for i, (sig, label, result, ax) in enumerate(zip(sigs, labels, results, axs)):

        # 시간 축 설정
        t = np.arange(len(sig)) / fs / 1e-6
        ax[0].plot(t, sig.real, 'c')
        ax[0].plot(t, sig.imag, 'm')

        ax[0].set_xlim([t[0], t[-1]])
        ax[0].set_xticks(np.arange(0, t[-1], 20))
        ax[0].set_title(f'Truth = {int(label)}\nResult = {result}')
        ax[0].set_xlabel('Time (us)')

        # 스펙트로그램
        ax[1].specgram(sig, NFFT=1024, Fs=fs / 1e6, noverlap=512, mode='psd', scale='dB', vmin=-100, vmax=-20)
        ax[1].set_xlim([t[0], t[-1]])
        ax[1].set_xticks(np.arange(0, t[-1], 20))
        ax[1].set_xlabel('Time (us)')

        # 파워 스펙트럼 밀도(PSD)
        nfft = len(sig)
        f = (np.arange(0, fs, fs / nfft) - (fs / 2) + (fs / nfft)) / 1e6
        y0 = np.fft.fftshift(np.fft.fft(sig))
        y = 10 * np.log10((y0 * y0.conjugate()).real / (nfft ** 2))
        ax[2].plot(f, y)
        ax[2].set_xlabel('Freq (MHz)')
        ax[2].set_ylim([-70, 30])

        if i > 0:
            [ax_row.set_yticks([]) for ax_row in ax]

    save_plot_name = model_save_folder / 'saved_model_plot.png'
    plt.savefig(str(save_plot_name))
    print(f'Output plot saved to: {save_plot_name}')

    return results

if __name__ == '__main__':
    _default_data_folder = '/data'
    infer(_default_data_folder)

