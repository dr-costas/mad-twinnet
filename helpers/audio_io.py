#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""This module is the `audio_io` and provides basic functionality for\
reading and writing audio files.

Reading and writing is supported for bit widths supported either by\
:mod:`wave` or :mod:`scipy.io.wavfile`. This module is a refactored\
version of parts of the code that can be found at\
`S. Mimilakis GitHub Repo <https://github.com/Js-Mim/mss_pytorch>`_.
"""

import wave

import numpy as np
from scipy.io.wavfile import write, read

__author__ = ['Konstantinos Drossos -- TUT', 'Stelios Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['wav_read', 'wav_write']

_normFact = {
    'int8': (2 ** 7) - 1,
    'int16': (2 ** 15) - 1,
    'int24': (2 ** 23) - 1,
    'int32': (2 ** 31) - 1,
    'int64': (2 ** 63) - 1,
    'float32': 1.0,
    'float64': 1.0
}


def wav_read(file_name, mono=False):
    """Reads a wav file and returns it data. If `mono` is \
    set to true, the returned audio data are monophonic.

    :param file_name: The file name of the wav file.
    :type file_name: str
    :param mono: Get mono version.
    :type mono: bool
    :return: The data and the sample rate.
    :rtype: (numpy.core.multiarray.ndarray, int)
    """
    try:
        samples, sample_rate = _load_wav_with_wave(file_name)
        sample_width = wave.open(file_name).getsampwidth()

        if sample_width == 1:
            # 8 bit case
            samples = (samples.astype(float) / _normFact['int8']) - 1.0
        elif sample_width == 2:
            # 16 bit case
            samples = samples.astype(float) / _normFact['int16']
        elif sample_width == 3:
            # 24 bit case
            samples = samples.astype(float) / _normFact['int24']
    except Exception:
        # 32 bit case
        samples, sample_rate = _load_wav_with_scipy(file_name)

    if samples.shape[1] == 1:
        samples = samples.squeeze(-1)

    # mono conversion
    if mono and samples.ndim == 2:
        samples = (samples[:, 0] + samples[:, 1]) * 0.5

    return samples, sample_rate


def wav_write(y, sampling_rate, nb_bits, file_name):
    """Writes audio data as wav file, using :func:`scipy.io.wavfile.write`.

    :param y: The audio data.
    :type y: numpy.core.multiarray.ndarray
    :param sampling_rate: The sampling rate.
    :type sampling_rate: int
    :param nb_bits: The number of bits.
    :type nb_bits: int
    :param file_name: The file name.
    :type file_name: str
    :raises ValueError: When the number of bits are not 8 or >= 16.
    """
    x = None

    if nb_bits == 8:
        int_samples = (y + 1.0) * _normFact['int' + str(nb_bits)]
        x = np.int8(int_samples)
    elif nb_bits == 16:
        int_samples = y * _normFact['int' + str(nb_bits)]
        x = np.int16(int_samples)
    elif nb_bits > 16:
        x = y
    if x is not None:
        write(file_name, sampling_rate, x)
    else:
        raise ValueError('Could not handle {} number of bits'.format(nb_bits))


def _load_wav_with_wave(file_name):
    """Loads a wav file with the :mod:`wave` package. Used\
    for wav files with sample width of 24 bits.

    :param file_name: The full file name (extension included).
    :type file_name: str
    :return: The audio data and the sampling rate.
    :rtype: (numpy.core.multiarray.ndarray, int)
    """
    wav = wave.open(file_name)
    rate = wav.getframerate()
    nb_channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    nb_frames = wav.getnframes()
    data = wav.readframes(nb_frames)
    wav.close()
    array = _wav_to_array(nb_channels, sample_width, data)

    return array, rate


def _load_wav_with_scipy(file_name):
    """Loads a wav file with the :func:`scipy.io.wavfile.read` function.

    :param file_name: The full file name (extension included).
    :type file_name: str
    :return: The audio data and the sampling rate.
    :rtype: (numpy.core.multiarray.ndarray, int)
    """
    input_data = read(file_name)
    samples = input_data[1]
    sample_rate = input_data[0]

    return samples, sample_rate


def _wav_to_array(nb_channels, sample_width, data):
    """Converts audio data to numpy ndarray.

    :param nb_channels: The amount of channels.
    :type nb_channels: int
    :param sample_width: The sample width in bits.
    :type sample_width: int
    :param data: The data.
    :type data: bytes
    :return: The `data` audio data as numpy ndarray
    :rtype: numpy.core.multiarray.ndarray
    """
    num_samples, remainder = divmod(len(data), sample_width * nb_channels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         '`sample_width` * `num_channels`.')
    if sample_width > 4:
        raise ValueError('`sample_width` must not be greater than 4.')

    if sample_width == 3:
        a = np.empty((num_samples, nb_channels, 4), dtype=np.uint8)
        raw_bytes = np.fromstring(data, dtype=np.uint8)
        a[:, :, :sample_width] = raw_bytes.reshape(-1, nb_channels, sample_width)
        a[:, :, sample_width:] = (a[:, :, sample_width - 1:sample_width] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sample_width == 1 else 'i'
        a = np.fromstring(data, dtype='<%s%d' % (dt_char, sample_width))
        result = a.reshape(-1, nb_channels)

    return result

# EOF
