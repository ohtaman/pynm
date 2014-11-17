# -*- coding:utf-8 -*-

from collections import defaultdict
import random


class EpsilonGreedyAgent:
    def __init__(self, epsilon=0.01):
        self._epsilon = epsilon
        self._counts = defaultdict(int)
        self._sums = defaultdict(int)

    def choose(self, arms, features=None):
        if random.random() <= self._epsilon:
            return random.choice(arms)
        else:
            return max(arms, key=lamnda arm: self._expected(arm))

    def _expected(self, arm):
        if self._counts[arm] > 0:
            return self._sums[arm]/self._counts[arm]
        else:
            return 0

    def update(self, arm, reward, arms=None, features=None):
        self._counts[arm] += 1
        self._sums[arm] += reward
