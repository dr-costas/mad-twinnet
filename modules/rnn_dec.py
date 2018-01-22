#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN dec of the Masker.
"""

import torch
from torch import nn
from torch.autograd import Variable

__author__ = 'Konstantinos Drossos -- TUT, Stylianos Mimilakis -- Fraunhofer IDMT'
__docformat__ = 'reStructuredText'


class RNNDec(nn.Module):
    def __init__(self, input_dim, debug):
        """The RNN dec of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param debug: Flag to indicate debug
        :type debug: bool
        """
        super(RNNDec, self).__init__()

        self._input_dim = input_dim
        self.gru_dec = nn.GRUCell(self._input_dim, self._input_dim)

        self._debug = debug

        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """

        nn.init.orthogonal(self.gru_dec.weight_hh)
        nn.init.xavier_normal(self.gru_dec.weight_ih)

        self.gru_dec.bias_hh.data.zero_()
        self.gru_dec.bias_ih.data.zero_()

    def forward(self, h_enc):
        """The forward pass.

        :param h_enc: The output of the RNN encoder.
        :type h_enc: torch.autograd.variable.Variable
        :return: The output of the RNN dec (h_j_dec)
        :rtype: torch.autograd.variable.Variable
        """
        batch_size = h_enc.size()[0]
        seq_length = h_enc.size()[1]
        h_t_dec = Variable(torch.zeros(batch_size, self._input_dim))
        h_j_dec = Variable(torch.zeros(batch_size, seq_length, self._input_dim))

        if not self._debug and torch.has_cudnn:
            h_t_dec = h_t_dec.cuda()
            h_j_dec = h_j_dec.cuda()

        for ts in range(seq_length):
            h_t_dec = self.gru_dec(h_enc[:, ts, :], h_t_dec)
            h_j_dec[:, ts, :] = h_t_dec

        return h_j_dec

# EOF
