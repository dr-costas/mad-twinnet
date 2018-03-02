#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage script.
"""

from __future__ import print_function

import argparse
import os
import time

import numpy as np
import torch
from torch.autograd import Variable

from helpers.data_feeder import data_feeder_testing, data_process_results_testing
from helpers.settings import debug, hyper_parameters, output_states_path, training_constants, \
    usage_output_string_per_example, usage_output_string_total
from modules import RNNEnc, RNNDec, FNNMasker, FNNDenoiser

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['use_me_process']


def use_me_process(sources_list, output_file_names):
    """The usage process.

    :param sources_list: The file names to be used.
    :type sources_list: list[str]
    :param output_file_names: The output file names to be used.
    :type output_file_names: list[list[str]]
    """

    print('\n-- Welcome to MaD TwinNet.')
    if debug:
        print('\n-- Cannot proceed in debug mode. Please set debug=False at the settings file.')
        print('-- Exiting.')
        exit(-1)
    print('-- Now I will extract the voice and the background music from the provided files')

    # Masker modules
    rnn_enc = RNNEnc(hyper_parameters['reduced_dim'], hyper_parameters['context_length'], debug)
    rnn_dec = RNNDec(hyper_parameters['rnn_enc_output_dim'], debug)
    fnn = FNNMasker(
        hyper_parameters['rnn_enc_output_dim'],
        hyper_parameters['original_input_dim'],
        hyper_parameters['context_length']
    )

    # Denoiser modules
    denoiser = FNNDenoiser(hyper_parameters['original_input_dim'])

    rnn_enc.load_state_dict(torch.load(output_states_path['rnn_enc']))
    rnn_dec.load_state_dict(torch.load(output_states_path['rnn_dec']))
    fnn.load_state_dict(torch.load(output_states_path['fnn']))
    denoiser.load_state_dict(torch.load(output_states_path['denoiser']))

    if not debug and torch.has_cudnn:
        rnn_enc = rnn_enc.cuda()
        rnn_dec = rnn_dec.cuda()
        fnn = fnn.cuda()
        denoiser = denoiser.cuda()

    testing_it = data_feeder_testing(
        window_size=hyper_parameters['window_size'], fft_size=hyper_parameters['fft_size'],
        hop_size=hyper_parameters['hop_size'], seq_length=hyper_parameters['seq_length'],
        context_length=hyper_parameters['context_length'], batch_size=1,
        debug=debug, sources_list=sources_list
    )

    print('-- Let\'s go!\n')
    total_time = 0

    for index, data in enumerate(testing_it()):

        s_time = time.time()

        mix, mix_magnitude, mix_phase, voice_true, bg_true = data

        voice_predicted = np.zeros(
            (
                mix_magnitude.shape[0],
                hyper_parameters['seq_length'] - hyper_parameters['context_length'] * 2,
                hyper_parameters['window_size']
            ),
            dtype=np.float32
        )

        for batch in range(int(mix_magnitude.shape[0] / training_constants['batch_size'])):
            b_start = batch * training_constants['batch_size']
            b_end = (batch + 1) * training_constants['batch_size']

            v_in = Variable(torch.from_numpy(mix_magnitude[b_start:b_end, :, :]))

            if not debug and torch.has_cudnn:
                v_in = v_in.cuda()

            tmp_voice_predicted = rnn_enc(v_in)
            tmp_voice_predicted = rnn_dec(tmp_voice_predicted)
            tmp_voice_predicted = fnn(tmp_voice_predicted, v_in)
            tmp_voice_predicted = denoiser(tmp_voice_predicted)

            voice_predicted[b_start:b_end, :, :] = tmp_voice_predicted.data.cpu().numpy()

        data_process_results_testing(
            index=index, voice_true=voice_true, bg_true=bg_true,
            voice_predicted=voice_predicted,
            window_size=hyper_parameters['window_size'], mix=mix, mix_magnitude=mix_magnitude,
            mix_phase=mix_phase, hop=hyper_parameters['hop_size'],
            context_length=hyper_parameters['context_length'],
            output_file_name=output_file_names[index]
        )

        e_time = time.time()

        print(usage_output_string_per_example.format(
            f=sources_list[index],
            t=e_time - s_time
        ))

        total_time += e_time - s_time

    print('\n-- Testing finished\n')
    print(usage_output_string_total.format(
        t=total_time
    ))
    print('-- That\'s all folks!')


def _make_target_file_names(sources_list):
    """Makes the target file names for the sources list.

    :param sources_list: The sources list.
    :type sources_list: list[str]
    :return: The target names.
    :rtype: list[list[str]]
    """
    targets_list = []

    for source in sources_list:
        f_name = os.path.splitext(source)[0]
        targets_list.append(['{}_voice.wav'.format(f_name), '{}_bg_music.wav'.format(f_name)])

    return targets_list


def _get_file_names_from_file(file_name):
    """Reads line by line a txt file and returns the contents.

    :param file_name: The file name of the txt file.
    :type file_name: str
    :return: The contents of the file, in a line-by-line fashion.
    :rtype: list[str]
    """
    with open(file_name) as f:
        return [line.strip() for line in f.readlines()]


def main():
    cmd_arg_parser = argparse.ArgumentParser(
        usage='python scripts/use_me [-w the_file.wav]|[-l the_files.txt]',
        description='Script to use the MaD TwinNet with your own files. Remember to set up properly'
                    'the PYTHONPATH environmental variable'
    )

    cmd_arg_parser.add_argument(
        '--input-wav', '-w', action='store', dest='input_wav', default='',
        help='Specify one wav file to be processed.'
    )

    cmd_arg_parser.add_argument(
        '--input-list', '-l', action='store', dest='input_list', default=[],
        help='Specify one txt file with each line to be one path for a wav file.'
    )

    cmd_args = cmd_arg_parser.parse_args()
    input_wav = cmd_args.input_wav
    input_list = cmd_args.input_list

    if (input_wav == '' and len(input_list) == 0) or (input_wav != '' and len(input_list) != 0):
        print('-- Please specify **either** a wav file (with -w) **or** give'
              'a txt file with file names in each line (with -l). ')
        print('-- Exiting.')
        exit(-1)

    if len(input_list) == 0:
        input_list = [input_wav]
    else:
        input_list = _get_file_names_from_file(input_list)

    use_me_process(
        sources_list=input_list,
        output_file_names=_make_target_file_names(input_list)
    )


if __name__ == '__main__':
    main()

# EOF
