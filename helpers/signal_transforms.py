#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import fftpack, signal

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['stft', 'i_stft', 'ideal_ratio_masking']

_eps = np.finfo(np.float32).tiny


def ideal_ratio_masking(mixture_in, magn_spectr_target, magn_spectr_residual):
    """Computation of Ideal Amplitude Ratio Mask. As appears in :\
    H Erdogan, John R. Hershey, Shinji Watanabe, and Jonathan Le Roux,\
    `Phase-sensitive and recognition-boosted speech separation using deep recurrent neural networks,`\
    in ICASSP 2015, Brisbane, April, 2015.

    :param mixture_in: The input mixture
    :type mixture_in: numpy.core.multiarray.ndarray
    :param magn_spectr_target: Magnitude Spectrogram of the target component
    :type magn_spectr_target: numpy.core.multiarray.ndarray
    :param magn_spectr_residual: Magnitude Spectrogram of the residual component
    :type magn_spectr_residual: numpy.core.multiarray.ndarray
    :return: Time-frequency gain values
    :rtype: numpy.core.multiarray.ndarray
    """
    mask = np.divide(magn_spectr_target, (_eps + magn_spectr_target + magn_spectr_residual))
    return np.multiply(mask, mixture_in)


def stft(x, windowing_func, fft_size, hop):
    """Short-time Fourier transform.

    :param x: Input time domain signal.
    :type x: numpy.core.multiarray.ndarray
    :param windowing_func: The windowing function to be used.
    :type windowing_func: numpy.core.multiarray.ndarray
    :param fft_size: The fft size in samples.
    :type fft_size: int
    :param hop: The hop size in samples.
    :type hop: int
    :return: The short-time Fourier transform of the input signal.
    :rtype: numpy.core.multiarray.ndarray
    """
    window_size = windowing_func.size

    x = np.append(np.zeros(3 * hop), x)
    x = np.append(x, np.zeros(3 * hop))

    p_in = 0
    p_end = x.size - window_size
    indx = 0

    if np.sum(windowing_func) != 0.:
        windowing_func = windowing_func / np.sqrt(fft_size)

    xm_x = np.zeros((int(len(x) / hop), int(fft_size / 2) + 1), dtype=np.float32)
    xp_x = np.zeros((int(len(x) / hop), int(fft_size / 2) + 1), dtype=np.float32)

    while p_in <= p_end:
        x_seg = x[p_in:p_in + window_size]

        mc_x, pc_x = _dft(x_seg, windowing_func, fft_size)

        xm_x[indx, :] = mc_x
        xp_x[indx, :] = pc_x

        p_in += hop
        indx += 1

    return xm_x, xp_x


def i_stft(magnitude_spect, phase, window_size, hop):
    """Short Time Fourier Transform synthesis of given magnitude and phase spectra,
    via iDFT.

    :param magnitude_spect: Magnitude spectrum.
    :type magnitude_spect: numpy.core.multiarray.ndarray
    :param phase: Phase spectrum.
    :type phase: numpy.core.multiarray.ndarray
    :param window_size: Synthesis window size in samples.
    :type window_size: int
    :param hop: Hop size in samples.
    :type hop: int
    :return: Synthesized time-domain signal.
    :rtype: numpy.core.multiarray.ndarray
    """
    rs = _gl_alg(window_size, hop, (window_size - 1) * 2)

    hw_1 = int(np.floor((window_size + 1) / 2))
    hw_2 = int(np.floor(window_size / 2))

    # Acquire the number of STFT frames
    nb_frames = magnitude_spect.shape[0]

    # Initialise output array with zeros
    time_domain_signal = np.zeros(nb_frames * hop + hw_1 + hw_2)

    # Initialise loop pointer
    pin = 0

    # Main Synthesis Loop
    for index in range(nb_frames):
        # Inverse Discrete Fourier Transform
        y_buf = _i_dft(magnitude_spect[index, :], phase[index, :], window_size)

        # Overlap and Add
        time_domain_signal[pin:pin + window_size] += y_buf * rs

        # Advance pointer
        pin += hop

    # Delete the extra zeros that the analysis had placed
    time_domain_signal = np.delete(time_domain_signal, range(3 * hop))
    time_domain_signal = np.delete(
        time_domain_signal,
        range(time_domain_signal.size - (3 * hop + 1),
              time_domain_signal.size)
    )

    return time_domain_signal


def _gl_alg(window_size, hop, fft_size=4096):
    """LSEE-MSTFT algorithm for computing the synthesis window.

    According to: Daniel W. Griffin and Jae S. Lim, `Signal estimation\
    from modified short-time Fourier transform,` IEEE Transactions on\
    Acoustics, Speech and Signal Processing, vol. 32, no. 2, pp. 236-243,\
    Apr 1984.

    :param window_size: Synthesis window size in samples. 
    :type window_size: int
    :param hop: Hop size in samples.
    :type hop: int
    :param fft_size: FTT size
    :type fft_size: int
    :return: The synthesized window
    :rtype: numpy.core.multiarray.ndarray
    """
    syn_w = signal.hamming(window_size) / np.sqrt(fft_size)
    syn_w_prod = syn_w ** 2.
    syn_w_prod.shape = (window_size, 1)
    redundancy = int(window_size / hop)
    env = np.zeros((window_size, 1))

    for k in range(-redundancy, redundancy + 1):
        env_ind = (hop * k)
        win_ind = np.arange(1, window_size + 1)
        env_ind += win_ind

        valid = np.where((env_ind > 0) & (env_ind <= window_size))
        env_ind = env_ind[valid] - 1
        win_ind = win_ind[valid] - 1
        env[env_ind] += syn_w_prod[win_ind]

    syn_w = syn_w / env[:, 0]

    return syn_w


def _dft(x, windowing_func, fft_size):
    """Discrete Fourier Transformation(Analysis) of a given real input signal.

    :param x: Input signal, in time domain
    :type x: numpy.core.multiarray.ndarray
    :param windowing_func: Windowing function
    :type windowing_func: numpy.core.multiarray.ndarray
    :param fft_size: FFT size in samples
    :type fft_size: int
    :return: Magnitude and phase of spectrum of `x`
    :rtype: numpy.core.multiarray.ndarray
    """
    half_n = int(fft_size / 2) + 1

    hw_1 = int(np.floor((windowing_func.size + 1) / 2))
    hw_2 = int(np.floor(windowing_func.size / 2))

    win_x = x * windowing_func

    fft_buffer = np.zeros(fft_size)
    fft_buffer[:hw_1] = win_x[hw_2:]
    fft_buffer[-hw_2:] = win_x[:hw_2]

    x = fftpack.fft(fft_buffer)

    magn_x = (np.abs(x[:half_n]))
    phase_x = (np.angle(x[:half_n]))

    return magn_x, phase_x


def _i_dft(magnitude_spect, phase, window_size):
    """Discrete Fourier Transformation(Synthesis) of a given spectral analysis
    via the :func:`scipy.fftpack.ifft` inverse FFT function.

    :param magnitude_spect: Magnitude spectrum.
    :type magnitude_spect: numpy.core.multiarray.ndarray
    :param phase: Phase spectrum.
    :type phase: numpy.core.multiarray.ndarray
    :param window_size: Synthesis window size.
    :type window_size: int
    :return: Time-domain signal.
    :rtype: numpy.core.multiarray.ndarray
    """
    # Get FFT Size
    fft_size = magnitude_spect.size
    fft_points = (fft_size - 1) * 2

    # Half of window size parameters
    hw_1 = int(np.floor((window_size + 1) / 2))
    hw_2 = int(np.floor(window_size / 2))

    # Initialise output spectrum with zeros
    tmp_spect = np.zeros(fft_points, dtype=complex)
    # Initialise output array with zeros
    time_domain_signal = np.zeros(window_size)

    # Compute complex spectrum(both sides) in two steps
    tmp_spect[0:fft_size] = magnitude_spect * np.exp(1j * phase)
    tmp_spect[fft_size:] = magnitude_spect[-2:0:-1] * np.exp(-1j * phase[-2:0:-1])

    # Perform the iDFT
    fft_buf = np.real(fftpack.ifft(tmp_spect))

    # Roll-back the zero-phase windowing technique
    time_domain_signal[:hw_2] = fft_buf[-hw_2:]
    time_domain_signal[hw_2:] = fft_buf[:hw_1]

    return time_domain_signal

# EOF
