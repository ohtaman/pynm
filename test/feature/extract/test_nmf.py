# -*- coding:utf-8 -*-

import logging
import math

from nose.tools import *
import numpy
import numpy.linalg

from pynm.feature.extract import nmf


logger = logging.getLogger(__name__)

@istest
def can_treat_matrix_without_errors():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix)
    ok_(True, 'Failed to decomposit a matrix')


@istest
def result_is_positive_matrix():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix)
    ok_(numpy.amin(w) >= 0, 'W of a matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'H of a matrix is not positive')

    matrix = numpy.zeros((4, 3))
    w, h = nmf.nmf(matrix)
    ok_(numpy.amin(w) >= 0, 'W of zero matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'W of zero matrix is not positive')


@istest
def can_reduce_dimension():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, 2)
    eq_(w.shape, (4, 2), 'dim(W) is not correct')
    eq_(h.shape, (2, 3), 'dim(W) is not correct')


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
    ok_(diff/100.0 < 0.12, 'NMF cannot apporximate a matrix (%s > 0.12)' % diff)

@istest
def can_approx_zero_matrix():
    matrix = numpy.zeros((4, 3))
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, max_iter=1000)
        diff += numpy.amax(w.dot(h))
    ok_(diff/100.0 < 0.12, 'NMF cannot apporximate zero matrix (%s > 0.12)' % diff)


@istest
def can_treat_matrix_without_errors_with_kl_divergent():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, distance="kl")
    ok_(True, 'Failed to decomposit a matrix')


@istest
def result_is_positive_matrix_with_kl_divergent():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, distance="kl")
    ok_(numpy.amin(w) >= 0, 'W of a matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'H of a matrix is not positive')

    matrix = numpy.zeros((4, 3))
    w, h = nmf.nmf(matrix, distance="kl")
    ok_(numpy.amin(w) >= 0, 'W of zero matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'W of zero matrix is not positive')


@istest
def can_reduce_dimension_with_kl_divergent():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, 2, distance="kl")
    eq_(w.shape, (4, 2), 'dim(W) is not correct')
    eq_(h.shape, (2, 3), 'dim(W) is not correct')


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
    ok_(diff/100.0 < 1, 'NMF cannot apporximate a matrix (%s > 1.)' % diff)


@istest
def can_approx_zero_matrix_with_kl_divergent():
    matrix = numpy.zeros((4, 3))
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, max_iter=1000, distance="kl")
        diff += numpy.amax(w.dot(h))
    ok_(diff/100.0 < 0.2, 'NMF cannot apporoximate zero matrix (%s > 0.2)' % diff)



@istest
def result_is_positive_matrix_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, init=nmf.random_init)
    ok_(numpy.amin(w) >= 0, 'W of a matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'H of a matrix is not positive')

    matrix = numpy.zeros((4, 3))
    w, h = nmf.nmf(matrix, init=nmf.random_init)
    ok_(numpy.amin(w) >= 0, 'W of zero matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'W of zero matrix is not positive')


@istest
def can_reduce_dimension_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, 2, init=nmf.random_init)
    eq_(w.shape, (4, 2), 'dim(W) is not correct')
    eq_(h.shape, (2, 3), 'dim(W) is not correct')


@istest
def can_approx_original_matrix_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, init=nmf.random_init)
        diff += numpy.amax(abs(matrix - w.dot(h)))
    ok_(diff/100.0 < 0.12, 'NMF cannot apporximate a matrix (%s > 0.12)' % diff)


@istest
def can_approx_zero_matrix_with_random_init():
    matrix = numpy.zeros((4, 3))
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, init=nmf.random_init, max_iter=1000)
        diff += numpy.amax(w.dot(h))
    ok_(diff/100.0 < 0.2, 'NMF cannot apporoximate zero matrix (%s > 0.2)' % diff)


@istest
def can_treat_matrix_without_errors_with_kl_divergent_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, init=nmf.random_init, distance="kl")
    ok_(True, 'Failed to decomposit a matrix')


@istest
def result_is_positive_matrix_with_kl_divergent_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, init=nmf.random_init, distance="kl")
    ok_(numpy.amin(w) >= 0, 'W of a matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'H of a matrix is not positive')

    matrix = numpy.zeros((4, 3))
    w, h = nmf.nmf(matrix, init=nmf.random_init, distance="kl")
    ok_(numpy.amin(w) >= 0, 'W of zero matrix is not positive')
    ok_(numpy.amin(h) >= 0, 'W of zero matrix is not positive')


@istest
def can_reduce_dimension_with_kl_divergent_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, dim=2, init=nmf.random_init, distance="kl")
    eq_(w.shape, (4, 2), 'dim(W) is not correct')
    eq_(h.shape, (2, 3), 'dim(W) is not correct')


@istest
def can_approx_original_matrix_with_kl_divergent_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, init=nmf.random_init, distance="kl")
        diff += numpy.amax(abs(matrix - w.dot(h)))
    ok_(diff/100.0 < 1, 'NMF cannot apporximate a matrix (%s > 1.0)' % diff)

@istest
def can_approx_zero_matrix_with_kl_divergent_with_random_init():
    matrix = numpy.zeros((4, 3))
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, init=nmf.random_init, max_iter=1000, distance="kl")
        diff += numpy.amax(w.dot(h))
    ok_(diff/100.0 < 0.2, 'NMF cannot apporoximate zero matrix (%s > 0.2)' % diff)

@istest
def can_treat_matrix_without_errors_with_beta_divergence_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, init=nmf.random_init, distance="beta")
    ok_(True, 'Failed to decomposit a matrix')


@istest
def result_is_positive_matrix_with_beta_divergence_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, init=nmf.random_init, distance="beta")
    w_min = numpy.amin(w)
    h_min = numpy.amin(h)
    ok_(w_min >= 0, 'W of zero matrix is not positive (%s < 0)' % w_min)
    ok_(h_min >= 0, 'H of zero matrix is not positive (%s < 0)' % h_min)

    matrix = numpy.zeros((4, 3))
    w, h = nmf.nmf(matrix, init=nmf.random_init, distance="beta")
    w_min = numpy.amin(w)
    h_min = numpy.amin(h)
    ok_(w_min >= 0, 'W of zero matrix is not positive (%s < 0)' % w_min)
    ok_(h_min >= 0, 'H of zero matrix is not positive (%s < 0)' % h_min)


@istest
def can_reduce_dimension_with_beta_divergence_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    w, h = nmf.nmf(matrix, dim=2, init=nmf.random_init, distance="beta")
    eq_(w.shape, (4, 2), 'dim(W) is not correct')
    eq_(h.shape, (2, 3), 'dim(W) is not correct')


@istest
def can_approx_original_matrix_with_beta_divergence_with_random_init():
    matrix = numpy.array([[1, 2, 3],
                          [0, 1, 7],
                          [7, 8, 1],
                          [9, 0, 1]])
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, init=nmf.random_init, distance="beta")
        diff += numpy.amax(abs(matrix - w.dot(h)))
    ok_(diff/100.0 < 1, 'NMF cannot apporximate a matrix (%s > 1.0)' % diff)

@istest
def can_approx_zero_matrix_with_beta_divergence_with_random_init():
    matrix = numpy.zeros((4, 3))
    diff = 0.0
    for _ in range(100):
        w, h = nmf.nmf(matrix, init=nmf.random_init, max_iter=1000, distance="beta")
        diff += numpy.amax(w.dot(h))
    ok_(diff/100.0 < 0.2, 'NMF cannot apporoximate zero matrix (%s > 0.2)' % diff)


