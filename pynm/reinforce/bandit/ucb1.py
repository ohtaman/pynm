# -*- coding:utf-8 -*-

from collections import defaultdict
import math
import random
import sys


class UCB1Agent:
    def __init__(self, epsilon=0.001):
        self._epsilon = epsiron
        self._counts = defaultdict(int)
        self._sums = defaultdict(int)
        self._total_count = 0

    def choose(self, arms, features=None):
        if random.random() <= self._epsilon:
            return random.choice(arms)
        else:
            return max(arms, key=lamnda arm: self._expected(arm))

    def _score(self, arm):
        if self._counts[arm] == 0:
            return sys.maxint
        else:
            count = self._counts[arm]
            return (self._sums[arm]/count
                    + math.sqrt(2*math.log(self._total_count)/count))

    def update(self, arm, reward, arms=None, features=None):
        self._total_count += 1
        self._counts[arm] += 1
        self._sums[arm] += reward
