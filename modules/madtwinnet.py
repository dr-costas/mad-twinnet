#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.nn import Module

from modules import masker, twin_net, fnn_denoiser, affine_transform

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['MaDTwinNet']


class MaDTwinNet(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim,
                 original_input_dim, context_length):
        super(MaDTwinNet, self).__init__()

        self.masker = masker.Masker(
            rnn_enc_input_dim=rnn_enc_input_dim,
            rnn_dec_input_dim=rnn_dec_input_dim,
            context_length=context_length,
            original_input_dim=original_input_dim
        )

        self.twin_net = twin_net.TwinNet(
            rnn_dec_input_dim=rnn_dec_input_dim,
            original_input_dim=original_input_dim,
            context_length=context_length
        )

        self.denoiser = fnn_denoiser.FNNDenoiser(
            input_dim=original_input_dim
        )

        self.affine = affine_transform.AffineTransform(
            input_dim=rnn_dec_input_dim
        )

    def forward(self, x):
        m_out = self.masker(x)
        v_j_filt_prime_twin = self.twin_net(m_out.h_enc, x)
        affine = self.affine(m_out.h_dec)


def main():
    pass


if __name__ == '__main__':
    main()

# EOF
