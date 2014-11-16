# -*- coding:utf-8 -*-

import logging
import math

from nose.tools import *
import numpy

from pynm.metric import itml


logger = logging.getLogger(__name__)

def cos(a, b, metric=None):
    sum_a = sum(a)
    sum_b = sum(b)
    if 0 in (sum_a, sum_b):
        return 0
    if metric is not None:
        return numpy.dot(a, numpy.dot(metric, b))/math.sqrt(sum(a)*sum(b))
    else:
        return numpy.dot(a, b)/math.sqrt(sum(a)*sum(b))


def test_sample_from_pairs():
    pairs = [(0, 1, True), (0, 2, False),
             (1, 0, True), (1, 2, False),
             (2, 0, False), (2, 1, False)]
    numpy.random.seed(0)
    samples = itml.sample_from_pairs(pairs)
    for i, sample in enumerate(samples):
        if i > 10:
            break
        ok_(sample in pairs, 'sample [%s] is not in pairs [%s]' % (sample, pairs))


def test_sample_from_labels():
    labels = [1, 2, 2, 3, 3, 3]
    numpy.random.seed(0)
    samples = itml.sample_from_labels(labels)
    for i, sample in enumerate(samples):
        if i > 10:
            break
        label1 = labels[sample[0]]
        label2 = labels[sample[1]]
        eq_(sample[2], label1 == label2)


def test_learn_metric_gather_similar_pairs():
    test_data = [numpy.array((1, 1, 1)),
                 numpy.array((1, 0, 0)),
                 numpy.array((0, 1, 0)),
                 numpy.array((0, 0, 0))]
    pairs = [(0, 1, True),
             (0, 2, False),
             (0, 3, False),
             (1, 0, True),
             (1, 2, False),
             (1, 3, False),
             (2, 0, False),
             (2, 1, False),
             (2, 3, True),
             (3, 0, False),
             (3, 1, False),
             (3, 2, True)]
    numpy.random.seed(0)

    metric = itml.learn_metric(test_data, pairs=pairs)
    base1 = cos(test_data[0], test_data[1])
    base2 = cos(test_data[0], test_data[1], metric)

    ok_(cos(test_data[0], test_data[2], metric)/base2
        <= cos(test_data[0], test_data[2])/base1,
        'same labeled pairs shold be closer together.')
    ok_(cos(test_data[0], test_data[3], metric)/base2
        <= cos(test_data[0], test_data[3])/base1,
        'different labeled pairs shold be away from each other.')

    eig = numpy.linalg.eig(metric)
    ok_(min(eig[0]) >= 0, 'the metric must be (semi-)definite.')

