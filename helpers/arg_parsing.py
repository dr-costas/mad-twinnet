#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

__author__ = 'Konstantinos Drossos'
__docformat__ = 'reStructuredText'
__all__ = ['get_argument_parser']


def get_argument_parser():
    """Creates and return the CMD argument parser.

    :return: The CMD argument parser.
    :rtype: argparse.ArgumentParser
    """
    cmd_arg_parser = argparse.ArgumentParser(
        usage='python scripts/use_me [-w the_file.wav]|[-l the_files.txt]',
        description='Script to use the MaD TwinNet with your own files. Remember to set up properly'
                    'the PYTHONPATH environmental variable'
    )

    cmd_arg_parser.add_argument(
        '--input-wav', '-w', action='store', dest='input_wav', default='',
        help='Specify one wav file to be processed.'
    )

    cmd_arg_parser.add_argument(
        '--input-list', '-l', action='store', dest='input_list', default=[],
        help='Specify one txt file with each line to be one path for a wav file.'
    )

    return cmd_arg_parser

# EOF
