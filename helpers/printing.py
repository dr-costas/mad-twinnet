#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from contextlib import ContextDecorator
from datetime import datetime

__author__ = 'Konstantinos Drossos -- TUT'
__docformat__ = 'reStructuredText'
__all__ = [
    'print_msg', 'inform_about_device', 'print_date_and_time',
    'InformAboutProcess', 'print_intro_messages'
]


_time_f_spec = '7.2'
_acc_f_spec = '6.2'
_loss_f_spec = '7.3'
_epoch_f_spec = '4'


def _print_empty_lines(nb_lines=1):
    """Prints empty lines.

    :param nb_lines: The amount of lines.
    :type nb_lines: int
    """
    for _ in range(nb_lines):
        print_msg('', start='', end='\n')


def print_intro_messages(device):
    """Prints initial messages.

    :param device: The device to be used.
    :type device: str
    """
    print_date_and_time()
    print_msg(' ', start='')
    inform_about_device(device)
    print_msg(' ', start='')


def print_msg(the_msg, start='-- ', end='\n', flush=True):
    """Prints a message.

    :param the_msg: The message.
    :type the_msg: str
    :param start: Starting decoration.
    :type start: str
    :param end: Ending character.
    :type end: str
    :param flush: Flush buffer now?
    :type flush: bool
    """
    print('{}{}'.format(start, the_msg), end=end, flush=flush)


def inform_about_device(the_device):
    """Prints an informative message about the device that we are using.

    :param the_device: The device.
    :type the_device: str
    """
    print_msg('Using device: `{}`.'.format(the_device))


def print_date_and_time():
    """Prints the date and time of `now`.
    """
    print_msg(datetime.now().strftime('%Y-%m-%d %H:%M'), start='\n\n-- ', end='\n\n')


class InformAboutProcess(ContextDecorator):
    def __init__(self, starting_msg, ending_msg='done', start='-- ', end='\n'):
        """Context manager and decorator for informing about a process.

        :param starting_msg: The starting message, printed before the process starts.
        :type starting_msg: str
        :param ending_msg: The ending message, printed after process ends.
        :type ending_msg: str
        :param start: Starting decorator for the string to be printed.
        :type start: str
        :param end: Ending decorator for the string to be printed.
        :type end: str
        """
        super(InformAboutProcess, self).__init__()
        self.starting_msg = starting_msg
        self.ending_msg = ending_msg
        self.start_dec = start
        self.end_dec = end

    def __enter__(self):
        print_msg('{}... '.format(self.starting_msg), start=self.start_dec, end='')

    def __exit__(self, *exc_type):
        print_msg('{}.'.format(self.ending_msg), start='', end=self.end_dec)

# EOF
