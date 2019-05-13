#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN dec of the Masker.
"""

import torch
from torch.nn import Module, GRU
from torch.nn.init import xavier_normal_, \
    orthogonal_, constant_

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['RNNDec']


class RNNDec(Module):
    def __init__(self, input_dim):
        """The RNN dec of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(RNNDec, self).__init__()

        self._input_dim = input_dim
        self.gru_dec = GRU(
            input_size=self._input_dim,
            hidden_size=self._input_dim,
            num_layers=1, bias=True,
            batch_first=True, bidirectional=False
        )

        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """

        xavier_normal_(self.gru_dec.weight_ih_l0)
        orthogonal_(self.gru_dec.weight_hh_l0)

        constant_(self.gru_dec.bias_ih_l0, 0)
        constant_(self.gru_dec.bias_hh_l0, 0)

    def forward(self, h_enc):
        """The forward pass.

        :param h_enc: The output of the RNN encoder.
        :type h_enc: torch.Tensor
        :return: The output of the RNN dec (h_j_dec).
        :rtype: torch.Tensor
        """
        return self.gru_dec(h_enc)[0]

# EOF
