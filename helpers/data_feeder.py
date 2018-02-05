#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Data getting and feeding module.
"""

import os
from operator import itemgetter

import numpy as np
from mir_eval import separation as bss_eval
from numpy.lib import stride_tricks
from scipy.signal import hamming

from helpers.audio_io import wav_read, wav_write
from helpers.settings import dataset_paths, output_audio_paths, wav_quality
from helpers.signal_transforms import stft, i_stft, ideal_ratio_masking

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['data_feeder_training', 'data_feeder_testing', 'data_process_results_testing']

_get_me_the_metrics = itemgetter(0, 2)


def data_feeder_training(window_size, fft_size, hop_size, seq_length, context_length,
                         batch_size, files_per_pass, debug):
    """Provides an iterator over the training examples.

    :param window_size: The window size to be used for the time-frequency transformation.
    :type window_size: int
    :param fft_size: The size of the FFT in samples.
    :type fft_size: int
    :param hop_size: The hop size in samples.
    :type hop_size: int
    :param seq_length: The sequence length in frames.
    :type seq_length: int
    :param context_length: The context length in frames.
    :type context_length: int
    :param batch_size: The batch size.
    :type batch_size: int
    :param files_per_pass: How many files per pass.
    :type files_per_pass: int
    :param batch_size: The batch size.
    :type batch_size: int
    :param files_per_pass: How many files per pass.
    :type files_per_pass: int
    :param debug: A flag to indicate debug
    :type debug: bool
    :return: An iterator that will provide the input and target values.\
             The iterator yields (input, target) values.
    :rtype: callable
    """
    mixtures_list, sources_list = _get_files_lists('training')
    hamming_window = hamming(window_size, True)

    def epoch_it():
        for index in range(int(len(mixtures_list) / files_per_pass)):
            mix, voice_true = _get_data_training(
                current_set=index + 1, set_size=files_per_pass,
                mixtures_list=mixtures_list, sources_list=sources_list,
                window_values=hamming_window, fft_size=fft_size, hop=hop_size,
                seq_length=seq_length, context_length=context_length,
                batch_size=batch_size
            )

            shuffled_indices = np.random.permutation(mix.shape[0])

            mix = mix[shuffled_indices]
            voice_true = voice_true[shuffled_indices]

            for batch in range(int(mix.shape[0] / batch_size)):

                b_start = batch * batch_size
                b_end = (batch + 1) * batch_size

                mix_batch = mix[b_start:b_end, :, :]
                voice_true_batch = voice_true[b_start:b_end, context_length:-context_length, :]

                yield mix_batch, voice_true_batch

                if debug:
                    break

            if debug:
                break

    return epoch_it


def data_feeder_testing(window_size, fft_size, hop_size, seq_length, context_length,
                        batch_size, debug, sources_list=None):
    """Provides an iterator over the testing examples.

    :param window_size: The window size to be used for the time-frequency transformation.
    :type window_size: int
    :param fft_size: The size of the FFT in samples.
    :type fft_size: int
    :param hop_size: The hop size in samples.
    :type hop_size: int
    :param seq_length: The sequence length in frames.
    :type seq_length: int
    :param context_length: The context length in frames.
    :type context_length: int
    :param batch_size: The batch size.
    :type batch_size: int
    :param debug: A flag to indicate debug
    :type debug: bool
    :param sources_list: The file list provided for using the MaD-TwinNet.
    :type sources_list: list[str]
    :return: An iterator that will provide the input and target values.\
             The iterator yields (mix, mix magnitude, mix phase, voice true, bg true) values.
    :rtype: callable
    """
    if sources_list is None:
        usage_case = False
        sources_list = _get_files_lists('testing')[-1]
    else:
        usage_case = True
    hamming_window = hamming(window_size, True)

    def testing_it():

        for index in range(len(sources_list)):
            yield _get_data_testing(
                sources_parent_path=sources_list[index],
                window_values=hamming_window, fft_size=fft_size, hop=hop_size,
                seq_length=seq_length, context_length=context_length,
                batch_size=batch_size, usage_case=usage_case
            )

            if debug:
                break

    return testing_it


def data_process_results_testing(index, voice_true, bg_true, voice_predicted,
                                 window_size, mix, mix_magnitude, mix_phase, hop,
                                 context_length, output_file_name=None):
    """Calculates SDR and SIR and creates the resulting audio files.

    :param index: The index of the current source/track.
    :type index: int
    :param voice_true: The true voice.
    :type voice_true: numpy.core.multiarray.ndarray
    :param bg_true: The true background music.
    :type bg_true: numpy.core.multiarray.ndarray
    :param voice_predicted: The predicted voice.
    :type voice_predicted: numpy.core.multiarray.ndarray
    :param window_size: The window size in samples.
    :type window_size: int
    :param mix: The mixture.
    :type mix: numpy.core.multiarray.ndarray
    :param mix_magnitude: The mixture magnitude.
    :type mix_magnitude: numpy.core.multiarray.ndarray
    :param mix_phase: The mixture phase.
    :type mix_phase: numpy.core.multiarray.ndarray
    :param hop: The hop size in samples.
    :type hop: int
    :param context_length: The context length in frames.
    :type context_length: int
    :param output_file_name: The output file name for the predicted voice\
                             and background music. If this argument is not
                             None, then the function just synthesizes the
                             voice and the background music, and saves them.
    :type output_file_name: list[str] | None
    :return: The values of SDR and SIR for each of the frames in\
             the current track, for both voice and background music.
    :rtype: (list[numpy.core.multiarray.ndarray], list[numpy.core.multiarray.ndarray])
    """
    voice_predicted.shape = (voice_predicted.shape[0] * voice_predicted.shape[1], window_size)
    mix_magnitude, mix_phase = _context_based_reshaping(mix_magnitude, mix_phase, context_length, window_size)

    voice_hat = i_stft(voice_predicted, mix_phase, window_size, hop)

    # Removing the samples that no estimation exists
    mix = mix[context_length * hop:]
    if output_file_name is None:
        voice_true = voice_true[context_length * hop:]
        bg_true = bg_true[context_length * hop:]
        min_len = min(len(voice_true), len(voice_hat))
        example_index = index + 1
    else:
        voice_true = None
        bg_true = None
        example_index = None
        min_len = min(len(mix), len(voice_hat))

    # Background music estimation
    bg_hat = mix[:min_len] - voice_hat[:min_len]

    if output_file_name is None:
        voice_hat_path = output_audio_paths['voice_predicted'].format(p=example_index)
        bg_hat_path = output_audio_paths['bg_predicted'].format(p=example_index)
        wav_write(voice_true, file_name=output_audio_paths['voice_true'].format(p=example_index), **wav_quality)
        wav_write(bg_true, file_name=output_audio_paths['bg_true'].format(p=example_index), **wav_quality)
        wav_write(mix, file_name=output_audio_paths['mix'].format(p=example_index), **wav_quality)

        # Metrics calculation
        sdr, sir = _get_me_the_metrics(bss_eval.bss_eval_images_framewise(
            [voice_true[:min_len], bg_true[:min_len]],
            [voice_hat[:min_len], bg_hat[:min_len]]
        ))

    else:
        voice_hat_path = output_file_name[0]
        bg_hat_path = output_file_name[1]

        sdr = None
        sir = None

    wav_write(voice_hat, file_name=voice_hat_path, **wav_quality)
    wav_write(bg_hat, file_name=bg_hat_path, **wav_quality)

    return sdr, sir


def _get_files_lists(subset):
    """Getting the files lists.

    :param subset: The subset that we are interested in (i.e. training or testing).
    :type subset: str
    :return: The lists with the file paths of the files that we want to use.
    :rtype: (list[str], list[str])
    """
    specific_dir = 'Dev' if subset == 'training' else 'Test'
    mixtures_dir = os.path.join(dataset_paths['mixtures'], specific_dir)
    sources_dir = os.path.join(dataset_paths['sources'], specific_dir)

    mixtures_list = [os.path.join(mixtures_dir, file_path)
                     for file_path in sorted(os.listdir(mixtures_dir))]

    sources_list = [os.path.join(sources_dir, file_path)
                    for file_path in sorted(os.listdir(sources_dir))]

    return mixtures_list, sources_list


def _context_based_reshaping(mix, voice, context_length, window_size):
    """A helper function to reshape data according
        to the context frame.
    """
    mix = np.ascontiguousarray(mix[:, context_length:-context_length, :], dtype=np.float32)
    mix.shape = (mix.shape[0] * mix.shape[1], window_size)
    voice = np.ascontiguousarray(voice[:, context_length:-context_length, :], dtype=np.float32)
    voice.shape = (voice.shape[0] * voice.shape[1], window_size)

    return mix, voice


def _make_overlap_sequences(mixture, voice, bg, l_size, o_lap, b_size):
    """Makes the overlap sequences to be used for time-frequency transformation.

    :param mixture: The mixture signal
    :type mixture: numpy.core.multiarray.ndarray
    :param voice: The voice signal
    :type voice: numpy.core.multiarray.ndarray
    :param bg: The background signal
    :type bg: numpy.core.multiarray.ndarray
    :param l_size: The context length in frames
    :type l_size: int
    :param o_lap: The overlap in samples
    :type o_lap: int
    :param b_size: The batch size
    :type b_size: int
    :return: The overlapping sequences
    :rtype: numpy.core.multiarray.ndarray
    """
    trim_frame = mixture.shape[0] % (l_size - o_lap)
    trim_frame -= (l_size - o_lap)
    trim_frame = np.abs(trim_frame)

    if trim_frame != 0:
        mixture = np.pad(mixture, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        voice = np.pad(voice, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))
        bg = np.pad(bg, ((0, trim_frame), (0, 0)), 'constant', constant_values=(0, 0))

    mixture = stride_tricks.as_strided(
        mixture,
        shape=(int(mixture.shape[0] / (l_size - o_lap)), l_size, mixture.shape[1]),
        strides=(mixture.strides[0] * (l_size - o_lap), mixture.strides[0], mixture.strides[1])
    )
    mixture = mixture[:-1, :, :]

    voice = stride_tricks.as_strided(
        voice,
        shape=(int(voice.shape[0] / (l_size - o_lap)), l_size, voice.shape[1]),
        strides=(voice.strides[0] * (l_size - o_lap), voice.strides[0], voice.strides[1])
    )
    voice = voice[:-1, :, :]

    bg = stride_tricks.as_strided(
        bg,
        shape=(int(bg.shape[0] / (l_size - o_lap)), l_size, bg.shape[1]),
        strides=(bg.strides[0] * (l_size - o_lap), bg.strides[0], bg.strides[1])
    )
    bg = bg[:-1, :, :]

    b_trim_frame = (mixture.shape[0] % b_size)
    if b_trim_frame != 0:
        mixture = mixture[:-b_trim_frame, :, :]
        voice = voice[:-b_trim_frame, :, :]
        bg = bg[:-b_trim_frame, :, :]

    return mixture, voice, bg


def _get_data_training(current_set, set_size, mixtures_list, sources_list,
                       window_values, fft_size, hop, seq_length, context_length,
                       batch_size):
    """Gets the actual input and output data for training.

    :param current_set: The current set of files that we are now looking.
    :type current_set: int
    :param set_size: The size of the sets that we consider.
    :type set_size: int
    :param mixtures_list: A list with the paths of the mixtures.
    :type mixtures_list: list[str]
    :param sources_list: A list with the paths of the source.
    :type sources_list: list[str]
    :param window_values: The values of the windowing function that we will use.
    :type window_values: numpy.core.multiarray.ndarray
    :param fft_size: The size of the FFT in samples.
    :type fft_size: int
    :param hop: The hop size in samples.
    :type hop: int
    :param seq_length: The sequence length in frames.
    :type seq_length: int
    :param context_length: The context length in frames.
    :type context_length: int
    :param batch_size: The batch size.
    :type batch_size: int
    :return: The actual input and target value.
    :rtype: numpy.core.multiarray.ndarray
    """
    m_list = mixtures_list[(current_set - 1) * set_size: current_set * set_size]
    s_list = sources_list[(current_set - 1) * set_size: current_set * set_size]

    ms_train, vs_train = None, None

    for index in range(len(m_list)):
        mix = wav_read(os.path.join(m_list[index], 'mixture.wav'), mono=False)[0]
        vox = wav_read(os.path.join(s_list[index], 'vocals.wav'), mono=False)[0]

        ms_seg = stft(
            0.5 * np.sum(mix, axis=-1), window_values,
            fft_size, hop
        )[0][3:-3, :]
        vs_seg = stft(
            0.5 * np.sum(vox, axis=-1), window_values,
            fft_size, hop
        )[0][3:-3, :]

        if index == 0:
            ms_train = ms_seg
            vs_train = vs_seg
        else:
            ms_train = np.vstack((ms_train, ms_seg))
            vs_train = np.vstack((vs_train, vs_seg))

    vs_train = ideal_ratio_masking(ms_train, vs_train, ms_train) * 2.
    vs_train = np.clip(vs_train, a_min=0., a_max=1.)

    ms_train = np.clip(ms_train, a_min=0., a_max=1.)
    ms_train, vs_train, _ = _make_overlap_sequences(
        ms_train, vs_train, ms_train,
        seq_length, context_length * 2, batch_size
    )

    return ms_train, vs_train


def _get_data_testing(sources_parent_path, window_values, fft_size, hop,
                      seq_length, context_length, batch_size, usage_case):
    """Gets the actual input and output data for testing.

    :param sources_parent_path: The parent path of the sources
    :type sources_parent_path: str
    :param window_values: The values of the windowing function that we will use.
    :type window_values: numpy.core.multiarray.ndarray
    :param fft_size: The size of the FFT in samples.
    :type fft_size: int
    :param hop: The hop size in samples.
    :type hop: int
    :param seq_length: The sequence length in frames.
    :type seq_length: int
    :param context_length: The context length in frames.
    :type context_length: int
    :param batch_size: The batch size.
    :type batch_size: int
    :param usage_case: Flag to indicate that currently we are just using it.
    :type usage_case: bool
    :return: The actual input and target value.
    :rtype: numpy.core.multiarray.ndarray
    """
    if not usage_case:
        bass = wav_read(os.path.join(sources_parent_path, 'bass.wav'), mono=False)[0]
        drums = wav_read(os.path.join(sources_parent_path, 'drums.wav'), mono=False)[0]
        others = wav_read(os.path.join(sources_parent_path, 'other.wav'), mono=False)[0]
        voice = wav_read(os.path.join(sources_parent_path, 'vocals.wav'), mono=False)[0]

        bg_true = np.sum(bass + drums + others, axis=-1) * 0.5
        voice_true = np.sum(voice, axis=-1) * 0.5
        mix = np.sum(bass + drums + others + voice, axis=-1) * 0.5
    else:
        mix = wav_read(sources_parent_path, mono=True)[0]
        voice_true = None
        bg_true = None

    mix_magnitude, mix_phase = stft(mix, window_values, fft_size, hop)

    # Data reshaping (magnitude and phase)
    mix_magnitude, mix_phase, _ = _make_overlap_sequences(
        mix_magnitude, mix_phase, mix_phase,
        seq_length, context_length * 2, batch_size)

    return mix, mix_magnitude, mix_phase, voice_true, bg_true

# EOF
