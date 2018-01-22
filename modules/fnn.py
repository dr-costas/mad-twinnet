#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The FNN of the Masker.
"""

import torch
from torch import nn

__author__ = 'Konstantinos Drossos -- TUT, Stylianos Mimilakis -- Fraunhofer IDMT'
__docformat__ = 'reStructuredText'


class FNNMasker(nn.Module):
    def __init__(self, input_dim, output_dim, context_length):
        """The FNN of the Masker.

        :param input_dim: The input dimensionality
        :type input_dim: int
        :param output_dim: The output dimensionality
        :type output_dim: int
        :param context_length: The context length
        :type context_length: int
        """

        super(FNNMasker, self).__init__()

        self._input_dim = input_dim
        self._output_dim = output_dim
        self._context_length = context_length

        self.linear_layer = nn.Linear(self._input_dim, self._output_dim)
        self.relu = nn.ReLU()

        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """
        nn.init.xavier_normal(self.linear_layer.weight)
        self.linear_layer.bias.data.zero_()

    def forward(self, h_j_dec, v_in):
        """Forward pass.

        :param h_j_dec: The output from the RNN decoder
        :type h_j_dec: torch.autograd.variable.Variable
        :param v_in: The original magnitude spectrogram input
        :type v_in: numpy.core.multiarray.ndarray
        :return: The output of the AffineTransform of the masker
        :rtype: torch.autograd.variable.Variable
        """
        v_in_prime = v_in[:, self._context_length:-self._context_length, :]
        m_j = self.relu(self.linear_layer(h_j_dec))
        v_j_filt_prime = torch.mul(m_j, v_in_prime)

        return v_j_filt_prime

# EOF
