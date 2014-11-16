# -*- coding:utf-8 -*-

from collections import defaultdict

from numpy import random


class ThompsonAgent:
    def __init__(self, epsilon=0.001):
        self._epsilon = epsiron
        self._successes = defaultdict(int)
        self._faileds = defaultdict(int)

    def choose(self, arms, features=None):
        return max(arms, key=lambda arm: self._score(arm))

    def _score(self, arm):
        return random.beta(self._successes[arm] + 1, self._faileds[arm] + 1)

    def update(self, arm, reward, arms=None, features=None):
        if reward > 0:
            self._successes[arm] += 1
        else:
            self._faileds[arms] += 1
