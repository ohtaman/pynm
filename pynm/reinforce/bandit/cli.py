#!/usr/bin/env python
# -*- coding:utf-8 -*-

import argparse
import json
import sys


def parse_args(args):
    parser = argparse.ArgumentParser(
        description='pynm Bandit Agent'
    )
    parser.add_argument('-i',
                        '--input',
                        default=None,
                        type=str,
                        metavar='FILE',
                        help='input file')
    parser.add_argument('-o',
                        '--output',
                        default=None,
                        type=str,
                        metavar='FILE',
                        help='output file')
    return parser.parse_args(args)


def main(argv=sys.argv):
    args = parse_args(argv[1:])
    input_ = open(args.input) if args.input is not None else sys.stdin
    output_ = open(args.output, 'w') if args.output is not None else sys.stdout

    while line in input_:
        data = json.loads(line)
        if data['type'] == 'choose':
            result = agent.choose(data)
        elif data['type'] == 'update':
            result = agent.update(data)
        output_.write('%s\n' % json.dumps(result, ensure_ascii=False))

    return 0

if __name__ == '__main__':
    exit(main())
