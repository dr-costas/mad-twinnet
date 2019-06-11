#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Testing process module.
"""
import pickle
import time
from functools import partial

import numpy as np
from torch import cuda, load, from_numpy, no_grad

from modules import MaD
from helpers import printing, data_feeder
from helpers.settings import debug, hyper_parameters,\
    output_states_path, training_constants, \
    testing_output_string_per_example,\
    metrics_paths, testing_output_string_all

__author__ = 'Konstantinos Drossos -- TAU, Stylianos Mimilakis -- Fraunhofer IDMT'
__docformat__ = 'reStructuredText'
__all__ = ['testing_process']


@no_grad()
def _testing_process(data, index, mad, device, seq_length,
                     context_length, window_size, batch_size,
                     hop_size):
    """The testing process over testing data.

    :param data: The testing data.
    :type data: numpy.ndarray
    :param index: The index of the testing data (used for\
                  calculating scores).
    :type index: int
    :param mad: The MaD system.
    :type mad: torch.nn.Module
    :param device: The device to be used.
    :type device: str
    :param seq_length: The sequence length used.
    :type seq_length: int
    :param context_length: The context length used.
    :type context_length: int
    :param window_size: The window size used.
    :type window_size: int
    :param batch_size: The batch size used.
    :type batch_size: int
    :param hop_size: The hop size used.
    :type hop_size: int
    :return: The SDR and SIR scores, and the time elapsed for\
             the process.
    :rtype: (numpy.ndarray, numpy.ndarray, float)
    """
    s_time = time.time()

    mix, mix_magnitude, mix_phase, voice_true, bg_true = data

    voice_predicted = np.zeros((
            mix_magnitude.shape[0],
            seq_length - context_length * 2,
            window_size), dtype=np.float32)

    for batch in range(int(mix_magnitude.shape[0] / batch_size)):
        b_start = batch * batch_size
        b_end = (batch + 1) * batch_size

        v_in = from_numpy(
            mix_magnitude[b_start:b_end, :, :]).to(device)

        voice_predicted[b_start:b_end, :, :] = mad(
            v_in).v_j_filt.cpu().numpy()

    tmp_sdr, tmp_sir = data_feeder.data_process_results_testing(
        index=index, voice_true=voice_true,
        bg_true=bg_true, voice_predicted=voice_predicted,
        window_size=window_size, mix=mix,
        mix_magnitude=mix_magnitude,
        mix_phase=mix_phase, hop=hop_size,
        context_length=context_length)

    time_elapsed = time.time() - s_time

    printing.print_msg(testing_output_string_per_example.format(
        e=index,
        sdr=np.median([i for i in tmp_sdr[0] if not np.isnan(i)]),
        sir=np.median([i for i in tmp_sir[0] if not np.isnan(i)]),
        t=time_elapsed
    ))

    return tmp_sdr, tmp_sir, time_elapsed


def testing_process():
    """The testing process.
    """
    # Check what device we'll be using
    device = 'cuda' if not debug and cuda.is_available() else 'cpu'

    # Inform about the device and time and date
    printing.print_intro_messages(device)
    printing.print_msg('Starting training process. '
                       'Debug mode: {}'.format(debug))

    # Set up MaD TwinNet
    with printing.InformAboutProcess('Setting up MaD TwinNet'):
        mad = MaD(
            rnn_enc_input_dim=hyper_parameters['reduced_dim'],
            rnn_dec_input_dim=hyper_parameters['rnn_enc_output_dim'],
            original_input_dim=hyper_parameters['original_input_dim'],
            context_length=hyper_parameters['context_length'])

    with printing.InformAboutProcess('Loading states'):
        mad.load_state_dict(load(output_states_path['mad']))
        mad = mad.to(device).eval()

    with printing.InformAboutProcess('Initializing data feeder'):
        testing_it = data_feeder.data_feeder_testing(
            window_size=hyper_parameters['window_size'],
            fft_size=hyper_parameters['fft_size'],
            hop_size=hyper_parameters['hop_size'],
            seq_length=hyper_parameters['seq_length'],
            context_length=hyper_parameters['context_length'],
            batch_size=1, debug=debug)

    p_testing = partial(
        _testing_process, mad=mad, device=device,
        seq_length=hyper_parameters['seq_length'],
        context_length=hyper_parameters['context_length'],
        window_size=hyper_parameters['window_size'],
        batch_size=training_constants['batch_size'],
        hop_size=hyper_parameters['hop_size'])

    printing.print_msg('Testing starts', end='\n\n')

    sdr, sir, total_time = [e for e in zip(*[
        i for index, data in enumerate(testing_it())
        for i in [p_testing(data, index)]])]

    total_time = sum(total_time)

    printing.print_msg('Testing finished', start='\n-- ', end='\n\n')
    printing.print_msg(testing_output_string_all.format(
        sdr=np.median([ii for i in sdr for ii in i[0] if not np.isnan(ii)]),
        sir=np.median([ii for i in sir for ii in i[0] if not np.isnan(ii)]),
        t=total_time), end='\n\n')

    with printing.InformAboutProcess('Saving results... '):
        with open(metrics_paths['sdr'], 'wb') as f:
            pickle.dump(sdr, f, protocol=2)
        with open(metrics_paths['sir'], 'wb') as f:
            pickle.dump(sir, f, protocol=2)

    printing.print_msg('That\'s all folks!')


def main():
    testing_process()


if __name__ == '__main__':
    main()

# EOF
