#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The Masker module.
"""

from collections import namedtuple

from torch.nn import Module

from modules import _rnn_enc, _rnn_dec, _fnn

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['Masker']


class Masker(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim,
                 context_length, original_input_dim):
        """The Masker module of the MaD TwinNet.

        :param rnn_enc_input_dim: The input dimensionality for\
                                  the RNN encoder.
        :type rnn_enc_input_dim: int
        :param rnn_dec_input_dim: The input dimensionality for\
                                  the RNN decoder.
        :type rnn_dec_input_dim: int
        :param context_length: The amount of time steps used for\
                               context length.
        :type context_length: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        """
        super(Masker, self).__init__()

        self.rnn_enc = _rnn_enc.RNNEnc(
            input_dim=rnn_enc_input_dim,
            context_length=context_length
        )
        self.rnn_dec = _rnn_dec.RNNDec(
            input_dim=rnn_dec_input_dim
        )

        self.fnn = _fnn.FNNMasker(
            input_dim=rnn_dec_input_dim,
            output_dim=original_input_dim,
            context_length=context_length
        )

        self.output = namedtuple(
            typename='masker_output',
            field_names=['h_enc', 'h_dec', 'v_j_filt_prime']
        )

    def forward(self, x):
        """Forward pass of the Masker.

        :param x: The input to the Masker.
        :type x: torch.Tensor
        :return: The outputs of the RNN encoder,\
                 RNN decoder, and the FNN.
        :rtype: collections.namedtuple
        """
        h_enc = self.rnn_enc(x)
        h_dec = self.rnn_dec(h_enc)
        return self.output(h_enc, h_dec, self.fnn(h_dec, x))

# EOF
