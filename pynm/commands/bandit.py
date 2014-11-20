# -*- coding:utf-8 -*-

import json
import logging
import sys

from pynm.reinforce.bandit.ucb1 import UCB1Agent
from pynm.reinforce.bandit.epsilon import EpsilonGreedyAgent
from pynm.reinforce.bandit.thompson import ThompsonAgent

class BanditCommand:
    name = 'bandit'
    help = 'Bandit Agent'

    algorithms = {'ucb1': UCB1Agent,
                  'epsilon': EpsilonGreedyAgent,
                  'thompson': ThompsonAgent}

    def build_arg_parser(self, parser):
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
        parser.add_argument('-a',
                            '--algorithm',
                            choices=self.algorithms.keys(),
                            default='thompson',
                            help='algorithm')

    def run(self, args):
        algorithm = self.algorithms[args.algorithm]()

        if args.output is not None:
            out_ = open(args.output, 'w')
        else:
            out_ = sys.stdout

        if args.input is not None:
            in_ = open(args.input)
        else:
            in_ = sys.stdin

        for line in in_:
            data = json.loads(line)
            if data['type'] == 'choose':
                arms = data['arms']
                features = data.get('features', None)
                result = algorithm.choose(arms, features)
                out_.write('%s\n' % result)
            elif data['type'] =='update':
                arm = data['arm']
                reward = data['reward']
                arms = data.get('arms', None)
                features = data.get('features', None)
                algorithm.update(arm, reward, arms, features)
            else:
                pass

        return 0
