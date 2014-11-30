# -*- coding:utf-8 -*-

import logging
import math

from nose.tools import *
import numpy
import numpy.linalg

from pynm.feature.select import nmf


logger = logging.getLogger(__name__)

@istest
def can_treat_matrix_without_errors():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix)
    ok_(True)


@istest
def result_is_positive_matrix():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix)
    ok_(numpy.amin(w) >= 0)
    ok_(numpy.amin(h) >= 0)

    matrix = numpy.zeros((4, 3))
    w, h = nmf.nmf(matrix)
    ok_(numpy.amin(w) >= 0)
    ok_(numpy.amin(h) >= 0)


@istest
def can_reduce_dimension():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, 2)
    eq_(w.shape, (4, 2))
    eq_(h.shape, (2, 3))


@istest
def can_approx_original_matrix():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix)
        diff += numpy.amax(abs(matrix - w.dot(h)))
        logger.info(diff)
    ok_(diff/100.0 < 0.12)

@istest
def can_approx_zero_matrix():
    matrix = numpy.zeros((4, 3))
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, max_iter=1000)
        diff += numpy.amax(w.dot(h))
        logger.info(diff)
    ok_(diff/100.0 < 0.12)


@istest
def can_treat_matrix_without_errors_with_kl_divergent():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, distance="kl")
    ok_(True)


@istest
def result_is_positive_matrix_with_kl_divergent():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, distance="kl")
    ok_(numpy.amin(w) >= 0)
    ok_(numpy.amin(h) >= 0)

    matrix = numpy.zeros((4, 3))
    w, h = nmf.nmf(matrix, distance="kl")
    ok_(numpy.amin(w) >= 0)
    ok_(numpy.amin(h) >= 0)


@istest
def can_reduce_dimension_with_kl_divergent():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, 2, distance="kl")
    eq_(w.shape, (4, 2))
    eq_(h.shape, (2, 3))


@istest
def can_approx_original_matrix_with_kl_divergent():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, distance="kl")
        diff += numpy.amax(abs(matrix - w.dot(h)))
        logger.info(diff)
    ok_(diff/100.0 < 1)

@istest
def can_approx_zero_matrix_with_kl_divergent():
    matrix = numpy.zeros((4, 3))
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, max_iter=1000, distance="kl")
        diff += numpy.amax(w.dot(h))
        logger.info(diff)
    ok_(diff/100.0 < 0.2)


