#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Testing process module.
"""

from __future__ import print_function

import pickle
import time

import numpy as np
import torch

from helpers.data_feeder import data_feeder_testing, data_process_results_testing
from helpers.settings import debug, hyper_parameters, output_states_path, training_constants, \
    testing_output_string_per_example, metrics_paths, testing_output_string_all
from modules import RNNEnc, RNNDec, FNNMasker, FNNDenoiser

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['testing_process']


def testing_process():
    """The testing process.
    """

    device = 'cuda' if not debug and torch.cuda.is_available() else 'cpu'

    print('\n-- Starting testing process. Debug mode: {}'.format(debug))
    print('-- Process on: {}'.format(device), end='\n\n')
    print('-- Setting up modules... ', end='')

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

    rnn_enc.load_state_dict(torch.load(output_states_path['rnn_enc'])).to(device)
    rnn_dec.load_state_dict(torch.load(output_states_path['rnn_dec'])).to(device)
    fnn.load_state_dict(torch.load(output_states_path['fnn'])).to(device)
    denoiser.load_state_dict(torch.load(output_states_path['denoiser'])).to(device)

    print('done.')

    testing_it = data_feeder_testing(
        window_size=hyper_parameters['window_size'], fft_size=hyper_parameters['fft_size'],
        hop_size=hyper_parameters['hop_size'], seq_length=hyper_parameters['seq_length'],
        context_length=hyper_parameters['context_length'], batch_size=1,
        debug=debug
    )

    print('-- Testing starts\n')

    sdr = []
    sir = []
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

            v_in = torch.from_numpy(mix_magnitude[b_start:b_end, :, :]).to(device)

            tmp_voice_predicted = rnn_enc(v_in)
            tmp_voice_predicted = rnn_dec(tmp_voice_predicted)
            tmp_voice_predicted = fnn(tmp_voice_predicted, v_in)
            tmp_voice_predicted = denoiser(tmp_voice_predicted)

            voice_predicted[b_start:b_end, :, :] = tmp_voice_predicted.data.cpu().numpy()

        tmp_sdr, tmp_sir = data_process_results_testing(
            index=index, voice_true=voice_true, bg_true=bg_true,
            voice_predicted=voice_predicted,
            window_size=hyper_parameters['window_size'], mix=mix, mix_magnitude=mix_magnitude,
            mix_phase=mix_phase, hop=hyper_parameters['hop_size'],
            context_length=hyper_parameters['context_length']
        )

        e_time = time.time()

        print(testing_output_string_per_example.format(
            e=index,
            sdr=np.median([i for i in tmp_sdr[0] if not np.isnan(i)]),
            sir=np.median([i for i in tmp_sir[0] if not np.isnan(i)]),
            t=e_time - s_time
        ))

        total_time += e_time - s_time

        sdr.append(tmp_sdr)
        sir.append(tmp_sir)

    print('\n-- Testing finished\n')
    print(testing_output_string_all.format(
        sdr=np.median([ii for i in sdr for ii in i[0] if not np.isnan(ii)]),
        sir=np.median([ii for i in sir for ii in i[0] if not np.isnan(ii)]),
        t=total_time
    ))

    print('\n-- Saving results... ', end='')

    with open(metrics_paths['sdr'], 'wb') as f:
        pickle.dump(sdr, f, protocol=2)

    with open(metrics_paths['sir'], 'wb') as f:
        pickle.dump(sir, f, protocol=2)

    print('done!')
    print('-- That\'s all folks!')


def main():
    testing_process()


if __name__ == '__main__':
    main()

# EOF
