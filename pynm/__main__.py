#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import logging
import sys

logger = logging.getLogger(__name__)
prog = 'pynm'
commands = ['hoge', 'hage']

def parse_args(args, prog=prog):
    parser = argparse.ArgumentParser(
        description='pynm Machine Leaner.',
        prog=prog
    )
    subparsers = parser.add_subparsers(help='command -h')
    for command in commands:
        command_parser = subparsers.add_parser(command, help='hoge')
        command_parser.add_argument('--fuge')
    return parser.parse_args(args)


def main(argv=sys.argv):
    args = parse_args(argv[1:])
    return 0

if __name__ == '__main__':
    exit(main())
