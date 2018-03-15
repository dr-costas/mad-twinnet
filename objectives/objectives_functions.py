#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Loss functions and regularizations
"""

from torch import norm

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['kullback_leibler', 'l2_loss', 'sparsity_penalty', 'l2_reg_squared']


def kullback_leibler(y_hat, y):
    """Generalized Kullback Leibler divergence.

    :param y_hat: The predicted distribution.
    :type y_hat: torch.autograd.variable.Variable
    :param y: The true distribution.
    :type y: torch.autograd.variable.Variable
    :return: The generalized Kullback Leibler divergence\
             between predicted and true distributions.
    :rtype: torch.autograd.variable.Variable
    """
    return (y * (y.add(1e-6).log() - y_hat.add(1e-6).log()) + (y_hat - y)).sum(dim=-1).mean()


def l2_loss(y_hat, y):
    """Measures the L2 loss between predicted and ground truth\
    values.

    :param y: The ground truth values.
    :type y: torch.autograd.variable.Variable
    :param y_hat: The predicted values.
    :type y_hat: torch.autograd.variable.Variable
    :return: The L2 loss.
    :rtype: torch.autograd.variable.Variable
    """
    return norm(y - y_hat, 2, dim=-1).mean()


def sparsity_penalty(weight_matrix):
    """Calculates the sparsity penalty for the FNN of the masker.

    :param weight_matrix: The weight matrix of the FNN of masker.
    :type weight_matrix: torch.autograd.variable.Variable
    :return: The sparsity penalty
    :rtype: torch.autograd.variable.Variable
    """
    return weight_matrix.data.diag().abs().sum()


def l2_reg_squared(weight_matrix):
    """Calculates the L2 regularization value for a weight matrix.

    :param weight_matrix: The weight matrix.
    :type weight_matrix: torch.autograd.variable.Variable
    :return: The L2 regularization value.
    :rtype: torch.autograd.variable.Variable
    """
    return weight_matrix.pow(2.).sum()

# EOF
