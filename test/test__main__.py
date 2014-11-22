# -*- coding:utf-8 -*-

import runpy
import os
import sys

from nose.tools import *

import pynm.__main__ as main_module


def setup():
    pass


def teardown():
    pass


def test_cli_needs_command_option():
    eq_(2, main_module.main(['pynm']))

def test_cli_can_show_help():
    eq_(0, main_module.main(['pynm', '-h']))

def test_cli_can_build_command_arg_parser():
    num = 1000
    class DummyCommand:
        name='test'
        help='help'
        def build_arg_parser(self, parser):
            parser.add_argument('--test', default=num)

        def run(self, args):
            return args.test

    parser = main_module.build_arg_parser(commands=[DummyCommand()])
    args = parser.parse_args(['test'])
    eq_(num, args.func(args))

