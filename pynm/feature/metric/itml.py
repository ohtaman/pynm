#!/usr/bin/env python
# -*- coding:utf-8 -*-

from collections import defaultdict
from random import choice
import sys

import numpy


def sample_from_pairs(pairs):
    num_pairs = len(pairs)
    while True:
        idx = numpy.random.randint(num_pairs)
        yield pairs[idx]


def sample_from_labels(labels, balance=False):
    num_labels = len(labels)
    if balance:
        maps = defaultdict(list)
        for idx, label in enumerate(labels):
            maps[label].append(idx)
        num_label_types = len(maps)
        items = maps.items()
        while True:
            label_type1 = numpy.random.choice(items)
            label_type2 = numpy.random.choice(items)
            idx1 = numpy.random.choice(label_type1[1])
            idx2 = numpy.random.choice(label_type2[1])
            yield (idx1, idx2, label_type1[0] == label_type2[0])
    else:
        while True:
            idx1 = numpy.random.randint(num_labels)
            idx2 = numpy.random.randint(num_labels)
            yield (idx1, idx2, labels[idx1] == labels[idx2])


def learn_metric(data,
                 labels=None,
                 pairs=None,
                 u=1,
                 l=1,
                 slack=1,
                 max_iter=1024,
                 balance=False,
                 is_sparse=False):
    if labels is not None:
        samples = sample_from_labels(labels)
    elif pairs is not None:
        samples = sample_from_pairs(pairs)
    else:
        raise RuntimeError('both target and pairs are None.')

    num_features = len(data[0])
    metric = numpy.identity(num_features)
    lambda_ = defaultdict(float)
    gsi_ = {}

    for cnt, sample in enumerate(samples):
        if cnt >= max_iter:
            break

        i = sample[0]
        j = sample[1]
        data1 = data[i]
        data2 = data[j]
        similar = sample[2]
        delta = 1 if similar else -1
        diff = numpy.array(data1) - numpy.array(data2)
        m_d = numpy.dot(metric, diff)
        p = numpy.dot(diff, m_d)
        if (i, j) in gsi_:
            gsi = gsi_[i, j]
        else:
            gsi = gsi_[i, j] = u if similar else l
        if p != 0.0:
            alpha = min(lambda_[i, j], delta*(1/p - slack/gsi)/2)
        else:
            alpha = lambda_[i, j]
        da = delta*alpha
        beta = da/(1-da*p)
        gsi_[i, j] *= slack/(slack + da*gsi)
        lambda_[i, j] -= alpha
        metric += beta * numpy.outer(m_d, m_d)
    return metric


def sqrt_matrix(matrix):
    u, s, v = numpy.linalg.svd(matrix)
    return numpy.dot(u, numpy.dot(numpy.diag(numpy.sqrt(s)), v))


def convert_data(metric, data):
    sqrt_metric = sqrt_matrix(metric)
    for a_data in data:
        yield numpy.dot(sqrt_metric, a_data)
