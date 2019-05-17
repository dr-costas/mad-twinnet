#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The MaD TwinNet system.
"""

from collections import namedtuple

from torch.nn import Module

from modules.mad import MaD
from modules._twin_net import TwinNet
from modules._affine_transform import AffineTransform

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['MaDTwinNet']


class MaDTwinNet(Module):

    def __init__(self, rnn_enc_input_dim, rnn_dec_input_dim,
                 original_input_dim, context_length):
        """The MaD TwinNet as a module.

        This class implements the MaD TwinNet as a module\
        and it is based on the separate modules of MaD and\
        TwinNet.

        :param rnn_enc_input_dim: The input dimensionality of\
                                  the RNN encoder.
        :type rnn_enc_input_dim: int
        :param rnn_dec_input_dim: The input dimensionality of\
                                  the RNN decoder.
        :type rnn_dec_input_dim: int
        :param original_input_dim: The original input dimensionality.
        :type original_input_dim: int
        :param context_length: The amount of time frames used as\
                               context.
        :type context_length: int
        """
        super(MaDTwinNet, self).__init__()

        self.mad = MaD(
            rnn_enc_input_dim=rnn_enc_input_dim,
            rnn_dec_input_dim=rnn_dec_input_dim,
            context_length=context_length,
            original_input_dim=original_input_dim
        )

        self.twin_net = TwinNet(
            rnn_dec_input_dim=rnn_dec_input_dim,
            original_input_dim=original_input_dim,
            context_length=context_length
        )

        self.affine = AffineTransform(
            input_dim=rnn_dec_input_dim
        )

        self.output = namedtuple(
            'mad_twin_net_output',
            [
                'v_j_filt_prime',
                'v_j_filt',
                'v_j_filt_prime_twin',
                'affine_output',
                'h_dec_twin'
            ]
        )

    def forward(self, x):
        """The forward pass of the MaD TwinNet.

        :param x: The input to the MaD TwinNet.
        :type x: torch.Tensor
        :return: The output of the MaD TwinNet. The\
                 fields of the named tuple are:
                   - `v_j_filt_prime`, the output of the Masker
                   - `v_j_filt`, the output of the Denoiser
                   - `v_j_filt_prime_twin`, the output of the\
                     TwinNet FNN
                   - `affine_output`, the output of the affine\
                     transform for the TwinNet regularization
                   - `h_dec_twin`, the output of the RNN of the\
                     TwinNet
        :rtype: collections.namedtuple[torch.Tensor, torch.Tensor\
                torch.Tensor, torch.Tensor, torch.Tensor]
        """
        # Masker pass
        mad_out = self.mad(x)

        # TwinNet pass
        twin_net_out = self.twin_net(mad_out.h_enc, x)

        # Twin net regularization
        affine = self.affine(mad_out.h_dec)

        return self.output(
            mad_out.v_j_filt_prime,
            mad_out.v_j_filt,
            twin_net_out.v_j_filt_prime_twin,
            affine,
            twin_net_out.h_dec_twin
        )

# EOF
