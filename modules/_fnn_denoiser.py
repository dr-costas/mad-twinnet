#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The FNN enc and FNN dec of the Denoiser.
"""

from torch.nn import Module, Linear
from torch.nn.functional import relu
from torch.nn.init import xavier_normal_, constant_

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['FNNDenoiser']


class FNNDenoiser(Module):

    def __init__(self, input_dim):
        """The FNN enc and FNN dec of the Denoiser.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(FNNDenoiser, self).__init__()

        self._input_dim = input_dim

        self.fnn_enc = Linear(
            in_features=self._input_dim,
            out_features=int(self._input_dim / 2),
            bias=True
        )

        self.fnn_dec = Linear(
            in_features=int(self._input_dim / 2),
            out_features=self._input_dim,
            bias=True
        )

        self.initialize_module()

    def initialize_module(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.fnn_enc.weight)
        constant_(self.fnn_enc.bias, 0)

        xavier_normal_(self.fnn_dec.weight)
        constant_(self.fnn_dec.bias, 0)

    def forward(self, v_j_filt_prime):
        """The forward pass.

        :param v_j_filt_prime: The output of the Masker.
        :type v_j_filt_prime: torch.Tensor
        :return: The output of the Denoiser.
        :rtype: torch.Tensor
        """
        fnn_enc_output = relu(self.fnn_enc(v_j_filt_prime))
        fnn_dec_output = relu(self.fnn_dec(fnn_enc_output))

        v_j_filt = fnn_dec_output.mul(v_j_filt_prime)

        return v_j_filt

# EOF
