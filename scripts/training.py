#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training process module.
"""

import time
from functools import partial

from torch import cuda, from_numpy
from torch import optim, nn, save

from helpers import data_feeder, printing
from helpers.settings import debug, hyper_parameters, training_constants, \
    training_output_string, output_states_path
from modules.madtwinnet import MaDTwinNet
from objectives import kullback_leibler as kl, l2_loss, sparsity_penalty, l2_reg_squared

__author__ = ['Konstantinos Drossos -- TAU', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['training_process']


def _one_epoch(module, epoch_it, solver, separation_loss, twin_reg_loss,
               reg_fnn_masker, reg_fnn_dec, device, epoch_index, lambda_l_twin,
               lambda_1, lambda_2, max_grad_norm):
    """One training epoch for MaD TwinNet.

    :param module: The module of MaD TwinNet.
    :type module: torch.nn.Module
    :param epoch_it: The data iterator for the epoch.
    :type epoch_it: callable
    :param solver: The optimizer to be used.
    :type solver: torch.optim.Optimizer
    :param separation_loss: The loss function used for\
                            the source separation.
    :type separation_loss: callable
    :param twin_reg_loss: The loss function used for the\
                          TwinNet regularization.
    :type twin_reg_loss: callable
    :param reg_fnn_masker: The weight regularization function\
                           for the FNN of the Masker.
    :type reg_fnn_masker: callable
    :param reg_fnn_dec: The weight regularization function\
                        for the FNN of the Denoiser.
    :type reg_fnn_dec: callable
    :param device: The device to be used.
    :type device: str
    :param epoch_index: The current epoch.
    :type epoch_index: int
    :param lambda_l_twin: The weight for the TwinNet loss.
    :type lambda_l_twin: float
    :param lambda_1: The weight for the `reg_fnn_masker`.
    :type lambda_1: float
    :param lambda_2: The weight for the `reg_fnn_dec`.
    :type lambda_2: float
    :param max_grad_norm: The maximum gradient norm for\
                          gradient norm clipping.
    :type max_grad_norm: float
    """
    def _training_iteration(_m, _data, _device, _solver, _sep_l, _reg_twin,
                            _reg_m, _reg_d, _lambda_l_twin, _lambda_1,
                            _lambda_2, _max_grad_norm):
        """One training iteration for the MaD TwinNet.

        :param _m: The module of MaD TwinNet.
        :type _m: torch.nn.Module
        :param _data: The data
        :type _data: numpy.ndarray
        :param _device: The device to be used.
        :type _device: str
        :param _solver: The optimizer to be used.
        :type _solver: torch.optim.Optimizer
        :param _sep_l: The loss function used for the\
                       source separation.
        :type _sep_l: callable
        :param _reg_twin: The loss function used for the\
                          TwinNet regularization.
        :type _reg_twin: callable
        :param _reg_m: The weight regularization function\
                       for the FNN of the Masker.
        :type _reg_m: callable
        :param _reg_d: The weight regularization function\
                       for the FNN of the Denoiser.
        :type _reg_d: callable
        :param _lambda_l_twin: The weight for the TwinNet loss.
        :type _lambda_l_twin: float
        :param _lambda_1: The weight for the `_reg_m`.
        :type _lambda_1: float
        :param _lambda_2: The weight for the `_reg_d`.
        :type _lambda_2: float
        :param _max_grad_norm: The maximum gradient norm for\
                               gradient norm clipping.
        :type _max_grad_norm: float
        :return: The losses for the iteration.
        :rtype: list[float]
        """
        # Get the data to torch and to the device used
        v_in, v_j = [from_numpy(_d).to(_device) for _d in _data]

        # Forward pass of the module
        output = _m(v_in)

        # Calculate losses
        l_m = _sep_l(output.v_j_filt_prime, v_j)
        l_d = _sep_l(output.v_j_filt, v_j)

        l_tw = _sep_l(output.v_j_filt_prime_twin, v_j).mul(_lambda_l_twin)
        l_twin = _reg_twin(output.affine_output, output.h_dec_twin.detach())

        w_reg_masker = _reg_m(_m.mad.masker.fnn.linear_layer.weight).mul(_lambda_1)
        w_reg_denoiser = _reg_d(_m.mad.denoiser.fnn_dec.weight).mul(_lambda_2)

        # Make MaD TwinNet objective
        loss = l_m.add(l_d).add(l_tw).add(l_twin).add(w_reg_masker).add(w_reg_denoiser)

        # Clear previous gradients
        _solver.zero_grad()

        # Backward pass
        loss.backward()

        # Gradient norm clipping
        nn.utils.clip_grad_norm_(_m.parameters(), max_norm=_max_grad_norm, norm_type=2)

        # Optimize
        _solver.step()

        return [l_m.item(), l_d.item(), l_tw.item(), l_twin.item()]

    # Log starting time
    time_start = time.time()

    # Do iteration over all batches
    iter_results = [
        _training_iteration(module, data, device, solver, separation_loss,
                            twin_reg_loss, reg_fnn_masker, reg_fnn_dec,
                            lambda_l_twin, lambda_1, lambda_2, max_grad_norm)
        for data in epoch_it()
    ]

    # Log ending time
    time_end = time.time()

    # Print to stdout
    printing.print_msg(training_output_string.format(
        ep=epoch_index,
        t=time_end - time_start,
        **{k: v for k, v in zip(['l_m', 'l_d', 'l_tw', 'l_twin'],
                                [sum(i)/len(iter_results)
                                 for i in zip(*iter_results)])
           }
    ))


def training_process():
    """The training process.
    """
    # Check what device we'll be using
    device = 'cuda' if not debug and cuda.is_available() else 'cpu'

    # Inform about the device and time and date
    printing.print_intro_messages(device)
    printing.print_msg('Starting training process. Debug mode: {}'.format(debug))

    # Set up MaD TwinNet
    with printing.InformAboutProcess('Setting up MaD TwinNet'):
        mad_twin_net = MaDTwinNet(
            rnn_enc_input_dim=hyper_parameters['reduced_dim'],
            rnn_dec_input_dim=hyper_parameters['rnn_enc_output_dim'],
            original_input_dim=hyper_parameters['original_input_dim'],
            context_length=hyper_parameters['context_length']
        ).to(device)

    # Get the optimizer
    with printing.InformAboutProcess('Setting up optimizer'):
        optimizer = optim.Adam(
            mad_twin_net.parameters(),
            lr=hyper_parameters['learning_rate']
        )

    # Create the data feeder
    with printing.InformAboutProcess('Initializing data feeder'):
        epoch_it = data_feeder.data_feeder_training(
            window_size=hyper_parameters['window_size'],
            fft_size=hyper_parameters['fft_size'],
            hop_size=hyper_parameters['hop_size'],
            seq_length=hyper_parameters['seq_length'],
            context_length=hyper_parameters['context_length'],
            batch_size=training_constants['batch_size'],
            files_per_pass=training_constants['files_per_pass'],
            debug=debug
        )

    # Inform about the future
    printing.print_msg('Training starts', end='\n\n')

    # Auxiliary function for aesthetics
    one_epoch = partial(
        _one_epoch, module=mad_twin_net,
        epoch_it=epoch_it, solver=optimizer,
        separation_loss=kl, twin_reg_loss=l2_loss,
        reg_fnn_masker=sparsity_penalty,
        reg_fnn_dec=l2_reg_squared, device=device,
        lambda_l_twin=hyper_parameters['lambda_l_twin'],
        lambda_1=hyper_parameters['lambda_1'],
        lambda_2=hyper_parameters['lambda_2'],
        max_grad_norm=hyper_parameters['max_grad_norm']
    )

    # Training
    [one_epoch(epoch_index=e) for e in range(training_constants['epochs'])]

    # Inform about the past
    printing.print_msg('Training done.', start='\n-- ')

    # Save the model
    with printing.InformAboutProcess('Saving model'):
        save(mad_twin_net.mad.state_dict(), output_states_path['mad'])

    # Say goodbye!
    printing.print_msg('That\'s all folks!')


def main():
    training_process()


if __name__ == '__main__':
    main()

# EOF
