#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Usage script.
"""
import time
from pathlib import Path

import numpy as np
import torch

from modules import MaD
from helpers import arg_parsing, printing, data_feeder
from helpers.settings import debug, hyper_parameters, \
    output_states_path, training_constants, \
    usage_output_string_per_example, usage_output_string_total

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['use_me_process']


def use_me_process(sources_list, output_file_names):
    """The usage process.

    :param sources_list: The file names to be used.
    :type sources_list: list[pathlib.Path]
    :param output_file_names: The output file names to be used.
    :type output_file_names: list[list[str]]
    """
    printing.print_msg('Welcome to MaD TwinNet.', end='\n\n')
    if debug:
        printing.print_msg('Cannot proceed in debug mode. '
                           'Please set `debug=False` at the settings '
                           'file.')
        printing.print_msg('Exiting.')
        exit(-1)
    printing.print_msg('Now I will extract the voice and the '
                       'background music from the provided files')

    device = 'cuda' if not debug and torch.cuda.is_available() else 'cpu'

    # MaD setting up
    mad = MaD(
        rnn_enc_input_dim=hyper_parameters['reduced_dim'],
        rnn_dec_input_dim=hyper_parameters['rnn_enc_output_dim'],
        original_input_dim=hyper_parameters['original_input_dim'],
        context_length=hyper_parameters['context_length'])

    mad.load_state_dict(torch.load(output_states_path['mad']))
    mad = mad.to(device).eval()

    testing_it = data_feeder.data_feeder_testing(
        window_size=hyper_parameters['window_size'], fft_size=hyper_parameters['fft_size'],
        hop_size=hyper_parameters['hop_size'], seq_length=hyper_parameters['seq_length'],
        context_length=hyper_parameters['context_length'], batch_size=1,
        debug=debug, sources_list=sources_list
    )

    printing.print_msg('Let\'s go!', end='\n\n')
    total_time = 0

    for index, data in enumerate(testing_it()):

        s_time = time.time()

        mix, mix_magnitude, mix_phase, voice_true, bg_true = data

        voice_predicted = np.zeros((
            mix_magnitude.shape[0],
            hyper_parameters['seq_length'] - hyper_parameters['context_length'] * 2,
            hyper_parameters['window_size']), dtype=np.float32)

        for batch in range(int(mix_magnitude.shape[0] / training_constants['batch_size'])):
            b_start = batch * training_constants['batch_size']
            b_end = (batch + 1) * training_constants['batch_size']

            v_in = torch.from_numpy(
                mix_magnitude[b_start:b_end, :, :]).to(device)

            voice_predicted[b_start:b_end, :, :] = mad(
                v_in).v_j_filt.cpu().numpy()

        data_feeder.data_process_results_testing(
            index=index, voice_true=voice_true, bg_true=bg_true,
            voice_predicted=voice_predicted,
            window_size=hyper_parameters['window_size'], mix=mix, mix_magnitude=mix_magnitude,
            mix_phase=mix_phase, hop=hyper_parameters['hop_size'],
            context_length=hyper_parameters['context_length'],
            output_file_name=output_file_names[index]
        )

        e_time = time.time()

        printing.print_msg(usage_output_string_per_example.format(
            f=sources_list[index],
            t=e_time - s_time))

        total_time += e_time - s_time

    printing.print_msg('MaDTwinNet finished')
    printing.print_msg(usage_output_string_total.format(t=total_time))
    printing.print_msg('That\'s all folks!')


def _make_target_file_names(sources_list):
    """Makes the target file names for the sources list.

    :param sources_list: The sources list.
    :type sources_list: list[patlib.Path]
    :return: The target names.
    :rtype: list[list[str]]
    """
    # targets_list = []
    #
    # for source in sources_list:
    #     f_name = os.path.splitext(source)[0]
    #     targets_list.append(['{}_voice.wav'.format(f_name),
    #                          '{}_bg_music.wav'.format(f_name)])
    #
    # return targets_list
    return [[
        '{}_voice.wav'.format(source.stem),
        '{}_bg_music.wav'.format(source.stem)]
        for source in sources_list
    ]


def _get_file_names_from_file(file_name):
    """Reads line by line a txt file and returns the contents.

    :param file_name: The file name of the txt file.
    :type file_name: str
    :return: The contents of the file, in a line-by-line fashion.
    :rtype: list[pathlib.Path]
    """
    with open(file_name) as f:
        return [Path(line.strip()) for line in f.readlines()]


def main():
    arg_parser = arg_parsing.get_argument_parser()
    cmd_args = arg_parser.parse_args()
    input_wav = cmd_args.input_wav
    input_list = cmd_args.input_list

    if (input_wav == '') != (len(input_list) != 0):
        printing.print_msg('Please specify **either** a wav file (with -w) '
                           '**or** give a txt file with file names in each '
                           'line (with -l). ')
        printing.print_msg('Exiting.')
        exit(-1)

    input_list = [Path(input_wav)] if len(input_list) == 0 \
        else _get_file_names_from_file(input_list)

    use_me_process(
        sources_list=input_list,
        output_file_names=[['{}_voice.wav'.format(source.stem),
                            '{}_bg_music.wav'.format(source.stem)]
                           for source in input_list])


if __name__ == '__main__':
    main()

# EOF
