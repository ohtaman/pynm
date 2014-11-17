# -*- coding:utf-8 -*-

from collections import defaultdict

from numpy import random


class ThompsonAgent:
    def __init__(self):
        self._succeeds = defaultdict(int)
        self._fails = defaultdict(int)

    def choose(self, arms, features=None):
        return max(arms, key=lambda arm: self._score(arm))

    def _score(self, arm):
        return random.beta(self._succeeds[arm] + 0.5, self._fails[arm] + 0.5)

    def update(self, arm, reward, arms=None, features=None):
        if reward > 0:
            self._succeeds[arm] += 1
        else:
            self._fails[arms] += 1
