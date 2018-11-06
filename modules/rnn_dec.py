#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN dec of the Masker.
"""

import torch
from torch.autograd import Variable
from torch.nn import Module, GRUCell
from torch.nn.init import xavier_normal_, orthogonal_

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['RNNDec']


class RNNDec(Module):
    def __init__(self, input_dim, debug):
        """The RNN dec of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param debug: Flag to indicate debug
        :type debug: bool
        """
        super(RNNDec, self).__init__()

        self._input_dim = input_dim
        self.gru_dec = GRUCell(self._input_dim, self._input_dim)

        self._debug = debug
        self._device = 'cuda' if not self._debug and torch.cuda.is_available() else 'cpu'

        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """

        xavier_normal_(self.gru_dec.weight_ih)
        orthogonal_(self.gru_dec.weight_hh)

        self.gru_dec.bias_ih.data.zero_()
        self.gru_dec.bias_hh.data.zero_()

    def forward(self, h_enc):
        """The forward pass.

        :param h_enc: The output of the RNN encoder.
        :type h_enc: torch.autograd.variable.Variable
        :return: The output of the RNN dec (h_j_dec)
        :rtype: torch.autograd.variable.Variable
        """
        batch_size = h_enc.size()[0]
        seq_length = h_enc.size()[1]
        h_t_dec = torch.zeros(batch_size, self._input_dim).to(self._device)
        h_j_dec = torch.zeros(batch_size, seq_length, self._input_dim).to(self._device)

        for ts in range(seq_length):
            h_t_dec = self.gru_dec(h_enc[:, ts, :], h_t_dec)
            h_j_dec[:, ts, :] = h_t_dec

        return h_j_dec

# EOF
