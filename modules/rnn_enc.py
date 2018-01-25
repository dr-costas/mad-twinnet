#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The RNN encoder of the Masker.
"""

from torch import has_cudnn as torch_has_cudnn, \
    zeros as torch_zeros, cat as torch_cat
from torch.autograd import Variable
from torch.nn import Module, GRUCell
from torch.nn.init import xavier_normal, orthogonal

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

        self.initialize_encoder()

    def initialize_encoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal(self.gru_enc_f.weight_ih)
        orthogonal(self.gru_enc_f.weight_hh)

        self.gru_enc_f.bias_ih.data.zero_()
        self.gru_enc_f.bias_hh.data.zero_()

        xavier_normal(self.gru_enc_b.weight_ih)
        orthogonal(self.gru_enc_b.weight_hh)

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

        h_t_f = Variable(torch_zeros(batch_size, self._input_dim))
        h_t_b = Variable(torch_zeros(batch_size, self._input_dim))
        h_enc = Variable(torch_zeros(
            batch_size,
            seq_length - (2 * self._context_length),
            2 * self._input_dim)
        )
        v_tr = v_in[:, :, :self._input_dim]

        if not self._debug and torch_has_cudnn:
            h_t_f = h_t_f.cuda()
            h_t_b = h_t_b.cuda()
            h_enc = h_enc.cuda()

        for t in range(seq_length):
            h_t_f = self.gru_enc_f((v_tr[:, t, :]), h_t_f)
            h_t_b = self.gru_enc_b((v_tr[:, seq_length - t - 1, :]), h_t_b)

            if self._context_length <= t < seq_length - self._context_length:
                h_t = torch_cat([
                    h_t_f + v_tr[:, t, :],
                    h_t_b + v_tr[:, seq_length - t - 1, :]
                ], dim=1)
                h_enc[:, t - self._context_length, :] = h_t

        return h_enc

# EOF
