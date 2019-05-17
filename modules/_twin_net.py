#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The TwinNet of the Masker.
"""

from collections import namedtuple

from torch.nn import Module

from modules import _twin_rnn_dec, _fnn

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['TwinNet']


class TwinNet(Module):

    def __init__(self, rnn_dec_input_dim, original_input_dim, context_length):
        super(TwinNet, self).__init__()

        self.rnn_dec = _twin_rnn_dec.TwinRNNDec(
            input_dim=rnn_dec_input_dim
        )

        self.fnn = _fnn.FNNMasker(
            input_dim=rnn_dec_input_dim,
            output_dim=original_input_dim,
            context_length=context_length
        )

        self.output = namedtuple(
            'twin_net_output',
            ['h_dec_twin', 'v_j_filt_prime_twin']
        )

    def forward(self, h_enc, x):
        """The forward pass of the TwinNet.

        :param h_enc: The input to the TwinNet.
        :type h_enc: torch.Tensor
        :param x: The original input to the non-twin\
                  counterpart.
        :type x: torch.Tensor
        :return: The output of the TwinNet.
        :rtype: torch.Tensor
        """
        h_dec_twin = self.rnn_dec(h_enc)
        return self.output(h_dec_twin, self.fnn(h_dec_twin, x))


# EOF
