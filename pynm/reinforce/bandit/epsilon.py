# -*- coding:utf-8 -*-

from collections import defaultdict
import numpy.random

class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.01, seed=None):
        self._epsilon = epsilon
        self._counts = defaultdict(int)
        self._sums = defaultdict(int)
        self._np_random = numpy.random.RandomState(seed)

    def choose(self, arms, features=None):
        if self._np_random.uniform() <= self._epsilon:
            return self._np_random.choice(arms)
        else:
            return max(arms, key=lambda arm: self._expected(arm))

    def _expected(self, arm):
        if self._counts[arm] > 0:
            return self._sums[arm]/self._counts[arm]
        else:
            return 0

    def update(self, arm, reward, arms=None, features=None):
        self._counts[arm] += 1
        self._sums[arm] += reward
