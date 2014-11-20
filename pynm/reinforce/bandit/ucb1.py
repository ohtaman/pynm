# -*- coding:utf-8 -*-

from collections import defaultdict
import math
import random
import sys


class UCB1Agent:
    def __init__(self, c=1):
        self._c = c
        self._counts = defaultdict(int)
        self._sums = defaultdict(int)
        self._total_count = 0

    def choose(self, arms, features=None):
        return max(arms, key=lambda arm: self._ucb(arm))

    def _ucb(self, arm):
        if self._counts[arm] == 0:
            return sys.maxsize
        else:
            count = self._counts[arm]
            return (self._sums[arm]/count
                    + self._c*math.sqrt(2*math.log(self._total_count)/count))

    def update(self, arm, reward, arms=None, features=None):
        self._total_count += 1
        self._counts[arm] += 1
        self._sums[arm] += reward
