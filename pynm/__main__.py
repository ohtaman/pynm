#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import logging
import sys

from pynm.commands import metric

logger = logging.getLogger(__name__)
prog = 'pynm'
commands = [metric.MetricCommand()]


def build_arg_parser(prog=prog, commands=commands):
    parser = argparse.ArgumentParser(
        description='pynm Machine Learning.',
        prog=prog
    )
    subparsers = parser.add_subparsers(title='commands')
    for command in commands:
        command_parser = subparsers.add_parser(command.name, help=command.help)
        command_parser.set_defaults(func=command.run)
        command.build_arg_parser(command_parser)
    return parser


def main(argv=sys.argv):
    parser = build_arg_parser()
    try:
        args = parser.parse_args(argv[1:])
    except SystemExit as e:
        return e.code
    if not hasattr(args, 'func'):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == '__main__':
    exit(main())
