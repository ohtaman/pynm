# -*- coding:utf-8 -*-

import logging
import math

from nose.tools import *
import numpy
import numpy.linalg

from pynm.feature.extract import svd


logger = logging.getLogger(__name__)

@istest
def can_treat_matrix_without_errors():
    matrix = numpy.array([[1, 2, 3],[0,1,7],[7,8,1],[9,0,1]])
    u, s, v = svd.svd(matrix)
    ok_(True, 'Failed to decomposit a matrix')


@istest
def can_approx_original_matrix():
    matrix = numpy.array([[1, 2, 3],[0,1,7],[7,8,1],[9,0,1]])
    u, s, v = svd.svd(matrix)
    diff = numpy.amax(abs(matrix -u.dot(numpy.diag(s)).dot(v)))
    ok_(diff < 0.001, ' SVD cannot approximate a matrix (%s > 0.001)' % diff)


@istest
def can_reduce_dimension():
    matrix = numpy.array([[1, 2, 3],[0,1,7],[7,8,1],[9,0,1]])
    u, s, v = svd.svd(matrix, 2, k=0)
    eq_(u.shape, (4, 2), 'dim(u) is not correct')
    eq_(s.shape, (2,), 'dim(s) is not correct')
    eq_(v.shape, (2, 3), 'dim(v) is not correct')

    u, s, v = svd.svd(matrix, 2, k=1)
    eq_(u.shape, (4, 2), 'dim(u) is not correct')
    eq_(s.shape, (2,), 'dim(s) is not correct')
    eq_(v.shape, (2, 3), 'dim(v) is not correct')


@istest
def reduced_version_can_approx_original_matrix():
    diff = 0.0
    matrix = numpy.array([[1, 2, 3],[0,1,7],[7,8,1],[9,0,1]])
    for _ in range(100):
        _, s, _ = svd.svd(matrix, 2, k=0)
        _, s_, _ = numpy.linalg.svd(matrix)
        diff += numpy.average(abs(s - s_[:2]))
    logger.info("svd average error is %s" % (diff/100.0))
    ok_(diff/100.0 < 1.2, 'SVD with dim reduction cannot aproximate a matrix (%s > 0.012)' % diff)
