#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN encoder of the Masker.
"""

import torch
from torch.nn import Module, GRU
from torch.nn.init import xavier_normal_, orthogonal_, constant_

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['RNNEnc']


class RNNEnc(Module):
    def __init__(self, input_dim, context_length):
        """The RNN encoder of the Masker.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        :param context_length: The context length.
        :type context_length: int
        """
        super(RNNEnc, self).__init__()

        self._input_dim = input_dim
        self._con_len = context_length

        self.gru_enc = GRU(
            input_size=self._input_dim,
            hidden_size=self._input_dim,
            num_layers=1, bias=True,
            batch_first=True, bidirectional=True
        )

        self.initialize_encoder()

    def initialize_encoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.gru_enc.weight_ih_l0)
        orthogonal_(self.gru_enc.weight_hh_l0)

        constant_(self.gru_enc.bias_ih_l0, 0)
        constant_(self.gru_enc.bias_hh_l0, 0)

        xavier_normal_(self.gru_enc.weight_ih_l0_reverse)
        orthogonal_(self.gru_enc.weight_hh_l0_reverse)

        constant_(self.gru_enc.bias_ih_l0_reverse, 0)
        constant_(self.gru_enc.bias_hh_l0_reverse, 0)

    def forward(self, v_in):
        """Forward pass.

        :param v_in: The input to the RNN encoder of the Masker.
        :type v_in: torch.Torch
        :return: The output of the Masker.
        :rtype: torch.Torch
        """
        # Trimming
        v_tr = v_in[:, :, :self._input_dim]

        # RNN encoder passing
        rnn_output = self.gru_enc(v_tr)[0]

        # Context dropping
        rnn_output = rnn_output[:, self._con_len:-self._con_len, :]

        # Residual connection and return
        return rnn_output + torch.cat([
            v_tr[:, self._con_len:-self._con_len, :, ],
            v_tr[:, self._con_len:-self._con_len, :, ].flip([1, 2])
        ], dim=-1)

# EOF
