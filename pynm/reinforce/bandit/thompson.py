# -*- coding:utf-8 -*-

from collections import defaultdict

import numpy

class ThompsonAgent:
    def __init__(self, seed=None):
        self._succeeds = defaultdict(int)
        self._fails = defaultdict(int)
        self._np_random = numpy.random.RandomState(seed)

    def choose(self, arms, features=None):
        return max(arms, key=lambda arm: self._score(arm))

    def _score(self, arm):
        return self._np_random.beta(
            self._succeeds[arm] + 0.5,
            self._fails[arm] + 0.5)

    def update(self, arm, reward, arms=None, features=None):
        if reward > 0:
            self._succeeds[arm] += 1
        else:
            self._fails[arms] += 1
