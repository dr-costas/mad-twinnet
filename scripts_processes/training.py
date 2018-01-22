#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training process module.
"""

import os
import time

import torch
from torch import optim
from torch.autograd import Variable

from helpers.data_feeder import training_data_feeder
from modules import RNNEnc, RNNDec, FNNMasker, FNNDenoiser, AffineTransform
from objectives import kullback_leibler as kl, l2_loss, sparsity_penalty, l2_reg

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['training_process']


def training_process():
    """The training process.
    """

    # Training constants
    debug = False
    print('\n-- Starting. Debug mode: {}'.format(debug))

    epochs = 2 if debug else 100
    batch_size = 16
    files_per_pass = 4

    # Hyper-parameters
    window_size = 2049
    fft_size = 4096
    hop_size = 384
    seq_length = 60
    context_length = 10
    reduced_dim = 744
    original_input_dim = 2049
    rnn_enc_output_dim = 2 * reduced_dim
    learning_rate = 1e-4
    max_grad_norm = .5
    lambda_l_twin = .5
    lambda_1 = 1e-2
    lambda_2 = 1e-4

    # Masker modules
    rnn_enc = RNNEnc(reduced_dim, context_length, debug)
    rnn_dec = RNNDec(rnn_enc_output_dim, debug)
    fnn = FNNMasker(rnn_enc_output_dim, original_input_dim, context_length)

    # Denoiser modules
    denoiser = FNNDenoiser(original_input_dim)

    # TwinNet regularization modules
    twin_net_rnn_dec = RNNDec(rnn_enc_output_dim, debug)
    twin_net_fnn_masker = FNNMasker(rnn_enc_output_dim, original_input_dim, context_length)
    affine_transform = AffineTransform(rnn_enc_output_dim)

    if not debug and torch.has_cudnn:
        rnn_enc = rnn_enc.cuda()
        rnn_dec = rnn_dec.cuda()
        fnn = fnn.cuda()
        denoiser = denoiser.cuda()
        twin_net_rnn_dec = twin_net_rnn_dec.cuda()
        twin_net_fnn_masker = twin_net_fnn_masker.cuda()
        affine_transform = affine_transform.cuda()

    print('-- Modules set up')

    # Objectives and penalties
    loss_masker = kl
    loss_denoiser = kl
    loss_twin = kl
    reg_twin = l2_loss
    reg_fnn_masker = sparsity_penalty
    reg_fnn_dec = l2_reg

    # Optimizer
    optimizer = optim.Adam(
        list(rnn_enc.parameters()) +
        list(rnn_dec.parameters()) +
        list(fnn.parameters()) +
        list(denoiser.parameters()) +
        list(twin_net_rnn_dec.parameters()) +
        list(twin_net_fnn_masker.parameters()) +
        list(affine_transform.parameters()),
        lr=learning_rate
    )

    print('-- Optimizer set up')

    # Initializing data feeder
    epoch_it = training_data_feeder(
        window_size=window_size,
        fft_size=fft_size,
        hop_size=hop_size,
        seq_length=seq_length,
        context_length=context_length,
        batch_size=batch_size,
        subset='training',
        files_per_pass=files_per_pass,
        debug=debug
    )

    print('-- Training starts\n')

    # Training loop starts
    for epoch in range(epochs):
        epoch_l_m = []
        epoch_l_d = []
        epoch_l_tw = []
        epoch_l_twin = []

        time_start = time.time()

        # Epoch loop
        for data in epoch_it():
            v_in = Variable(torch.from_numpy(data[0]))
            v_j = Variable(torch.from_numpy(data[1]))

            if not debug and torch.has_cudnn:
                v_in = v_in.cuda()
                v_j = v_j.cuda()

            # Masker pass
            h_enc = rnn_enc(v_in)
            h_dec = rnn_dec(h_enc)
            v_j_filt_prime = fnn(h_dec, v_in)

            # TwinNet pass
            h_dec_twin = twin_net_rnn_dec(h_enc)
            v_j_filt_prime_twin = twin_net_fnn_masker(h_dec_twin, v_in)

            # Twin net regularization
            affine_output = affine_transform(h_dec)

            # Denoiser pass
            v_j_filt = denoiser(v_j_filt_prime)

            optimizer.zero_grad()

            # Calculate losses
            l_m = loss_masker(v_j_filt_prime, v_j)
            l_d = loss_denoiser(v_j_filt, v_j)
            l_tw = loss_twin(v_j_filt_prime_twin, v_j)
            l_twin = reg_twin(affine_output, h_dec_twin)

            # Make MaD TwinNet objective
            loss = l_m + l_d + l_tw + (lambda_l_twin * l_twin) + \
                   (lambda_1 * reg_fnn_masker(fnn.linear_layer.weight)) + \
                   (lambda_2 * reg_fnn_dec(denoiser.fnn_dec.weight))

            # Backward pass
            loss.backward()

            # Gradient norm clipping
            torch.nn.utils.clip_grad_norm(
                list(rnn_enc.parameters()) +
                list(rnn_dec.parameters()) +
                list(fnn.parameters()) +
                list(denoiser.parameters()) +
                list(twin_net_rnn_dec.parameters()) +
                list(twin_net_fnn_masker.parameters()) +
                list(affine_transform.parameters()),
                max_norm=max_grad_norm, norm_type=2
            )

            # Optimize
            optimizer.step()

            # Log losses
            epoch_l_m.append(l_m.data[0])
            epoch_l_d.append(l_d.data[0])
            epoch_l_tw.append(l_tw.data[0])
            epoch_l_twin.append(l_twin.data[0])

        time_end = time.time() - time_start

        # Tell us what happened
        print(
            'Epoch: {ep:3d} Losses: -- Masker: {l_m:.4f} | Denoiser: {l_d:.4f} | '
            'Twin: {l_tw:.4f} | Twin reg.: {l_twin:.4f} | Time: {t:.2f} sec(s)'.format(
                ep=epoch,
                l_m=torch.mean(torch.FloatTensor(epoch_l_m)),
                l_d=torch.mean(torch.FloatTensor(epoch_l_d)),
                l_tw=torch.mean(torch.FloatTensor(epoch_l_tw)),
                l_twin=torch.mean(torch.FloatTensor(epoch_l_tw)),
                t=time_end
            ))

    # Kindly end and save the model
    print('\n-- Training done. Saving model')
    torch.save(rnn_enc.state_dict(), os.path.join('states', 'rnn_enc.pt'))
    torch.save(rnn_dec.state_dict(), os.path.join('states', 'rnn_dec.pt'))
    torch.save(fnn.state_dict(), os.path.join('states', 'fnn.pt'))
    torch.save(denoiser.state_dict(), os.path.join('states', 'denoiser.pt'))
    print('-- Model saved.')
    print('-- That\'s all folks!')


def main():
    training_process()


if __name__ == '__main__':
    main()

# EOF
