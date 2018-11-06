#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN encoder of the Masker.
"""

import torch
from torch.nn import Module, GRUCell
from torch.nn.init import xavier_normal_, orthogonal_

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['RNNEnc']


class RNNEnc(Module):
    def __init__(self, input_dim, context_length, debug):
        """The RNN encoder of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param context_length: The context length.
        :type context_length: int
        :param debug: Flag to indicate debug
        :type debug: bool
        """
        super(RNNEnc, self).__init__()

        self._input_dim = input_dim
        self._context_length = context_length

        self.gru_enc_f = GRUCell(self._input_dim, self._input_dim)
        self.gru_enc_b = GRUCell(self._input_dim, self._input_dim)

        self._debug = debug
        self._device = 'cuda' if not self._debug and torch.cuda.is_available() else 'cpu'

        self.initialize_encoder()

    def initialize_encoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.gru_enc_f.weight_ih)
        orthogonal_(self.gru_enc_f.weight_hh)

        self.gru_enc_f.bias_ih.data.zero_()
        self.gru_enc_f.bias_hh.data.zero_()

        xavier_normal_(self.gru_enc_b.weight_ih)
        orthogonal_(self.gru_enc_b.weight_hh)

        self.gru_enc_b.bias_ih.data.zero_()
        self.gru_enc_b.bias_hh.data.zero_()

    def forward(self, v_in):
        """Forward pass.

        :param v_in: The input to the RNN encoder of the Masker.
        :type v_in: numpy.core.multiarray.ndarray
        :return: The output of the RNN encoder of the Masker.
        :rtype: torch.autograd.variable.Variable
        """
        batch_size = v_in.size()[0]
        seq_length = v_in.size()[1]

        h_t_f = torch.zeros(batch_size, self._input_dim).to(self._device)
        h_t_b = torch.zeros(batch_size, self._input_dim).to(self._device)

        h_enc = torch.zeros(
            batch_size, seq_length - (2 * self._context_length), 2 * self._input_dim
        ).to(self._device)

        v_tr = v_in[:, :, :self._input_dim]

        for t in range(seq_length):
            h_t_f = self.gru_enc_f((v_tr[:, t, :]), h_t_f)
            h_t_b = self.gru_enc_b((v_tr[:, seq_length - t - 1, :]), h_t_b)

            if self._context_length <= t < seq_length - self._context_length:
                h_t = torch.cat([
                    h_t_f + v_tr[:, t, :],
                    h_t_b + v_tr[:, seq_length - t - 1, :]
                ], dim=1)
                h_enc[:, t - self._context_length, :] = h_t

        return h_enc

# EOF
