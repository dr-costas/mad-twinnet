#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The affine transform for the TwinNet regularization.
"""

from torch.nn import Module, Linear
from torch.nn.init import xavier_normal_, constant_

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['AffineTransform']


class AffineTransform(Module):
    def __init__(self, input_dim):
        """The affine transform for the TwinNet regularization.

        :param input_dim: The input dimensionality.
        :type input_dim: int
        """
        super(AffineTransform, self).__init__()

        self._input_dim = input_dim
        self.linear_layer = Linear(self._input_dim, self._input_dim)

        self.initialize_decoder()

    def initialize_decoder(self):
        """Manual weight/bias initialization.
        """
        xavier_normal_(self.linear_layer.weight)
        constant_(self.linear_layer.bias, 0)

    def forward(self, h_j_dec):
        """Forward pass.

        :param h_j_dec: The output from the RNN decoder.
        :type h_j_dec: torch.Tensor
        :return: The output of the affine transform.
        :rtype: torch.Tensor
        """
        return self.linear_layer(h_j_dec)

# EOF
