#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The settings of the modules and the process
"""

# imports
import os

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = [
    'debug',
    'dataset_paths',
    'output_audio_paths',
    'metrics_paths',
    'output_states_path',
    'training_output_string',
    'testing_output_string_per_example',
    'testing_output_string_all',
    'training_constants',
    'wav_quality',
    'hyper_parameters',
    'usage_output_string_per_example',
    'usage_output_string_total'
]


debug = False
_debug_suffix = '_debug' if debug else ''

# Paths
_dataset_parent_dir = 'dataset'
_outputs_path = 'outputs'
_states_path = os.path.join(_outputs_path, 'states')
_metrics_path = os.path.join(_outputs_path, 'metrics')
_audio_files_path = os.path.join(_outputs_path, 'audio_files')

dataset_paths = {
    'mixtures': os.path.join(_dataset_parent_dir, 'Mixtures'),
    'sources': os.path.join(_dataset_parent_dir, 'Sources')
}

output_audio_paths = {
    'voice_true': os.path.join(
        _audio_files_path,
        'test_example_{placeholder}_voice_true{d}.wav'.format(
            placeholder='{p:02d}', d=_debug_suffix)),
    'voice_predicted': os.path.join(
        _audio_files_path,
        'test_example_{placeholder}_voice_predicted{d}.wav'.format(
            placeholder='{p:02d}', d=_debug_suffix)),
    'bg_true': os.path.join(
        _audio_files_path,
        'test_example_{placeholder}_bg_true{d}.wav'.format(
            placeholder='{p:02d}', d=_debug_suffix)),
    'bg_predicted': os.path.join(
        _audio_files_path,
        'test_example_{placeholder}_bg_predicted{d}.wav'.format(
            placeholder='{p:02d}', d=_debug_suffix)),
    'mix': os.path.join(
        _audio_files_path,
        'test_example_{placeholder}_mix_true{d}.wav'.format(
            placeholder='{p:02d}', d=_debug_suffix))
}

metrics_paths = {
    'sdr': os.path.join(_metrics_path, 'sdr{}_p2.pckl'.format(_debug_suffix)),
    'sir': os.path.join(_metrics_path, 'sir{}_p2.pckl'.format(_debug_suffix))
}

output_states_path = {
    'rnn_enc': os.path.join(_states_path, 'rnn_enc{}.pt'.format(_debug_suffix)),
    'rnn_dec': os.path.join(_states_path, 'rnn_dec{}.pt'.format(_debug_suffix)),
    'fnn': os.path.join(_states_path, 'fnn{}.pt'.format(_debug_suffix)),
    'denoiser': os.path.join(_states_path, 'denoiser{}.pt'.format(_debug_suffix))
}

# Strings
training_output_string = 'Epoch: {ep:3d} Losses: -- ' \
                         'Masker:{l_m:6.4f} | Denoiser:{l_d:6.4f} | ' \
                         'Twin:{l_tw:6.4f} | Twin reg.:{l_twin:6.4f} | ' \
                         'Time:{t:6.2f} sec(s)'

testing_output_string_per_example = 'Example: {e:2d}, Median -- ' \
                                    'SDR:{sdr:6.2f} dB | SIR:{sir:6.2f} dB | ' \
                                    'Time:{t:6.2f} sec(s)'

testing_output_string_all = 'Median SDR:{sdr:6.2f} dB | ' \
                            'Median SIR:{sir:6.2f} dB | ' \
                            'Total time:{t:6.2f} sec(s)'

usage_output_string_per_example = '-- File {f} processed. Time: {t:6.2f} sec(s)'
usage_output_string_total = '-- All files processed. Total time: {t:6.2f} sec(s)'

# Process constants
training_constants = {
    'epochs': 2 if debug else 100,
    'batch_size': 16,
    'files_per_pass': 4
}

wav_quality = {'sampling_rate': 44100, 'nb_bits': 16}

# Hyper-parameters
hyper_parameters = {
    'window_size': 2049,
    'fft_size': 4096,
    'hop_size': 384,
    'seq_length': 60,
    'context_length': 10,
    'reduced_dim': 744,
    'original_input_dim': 2049,
    'learning_rate': 1e-4,
    'max_grad_norm': .5,
    'lambda_l_twin': .5,
    'lambda_1': 1e-2,
    'lambda_2': 1e-4
}
hyper_parameters.update({
    'rnn_enc_output_dim': 2 * hyper_parameters['reduced_dim']
})

# EOF
