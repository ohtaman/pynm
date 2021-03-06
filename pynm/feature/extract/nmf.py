# -*- coding:utf-8 -*-

import numpy
import numpy.random
import numpy.linalg

from . import svd


def svd_init(matrix, dim, seed=None):
    u, s, v = svd.svd(matrix, dim)
    ss = numpy.sqrt(numpy.diag(s))
    return numpy.maximum(0.001, u.dot(ss)), numpy.maximum(0.001, ss.dot(v))

def random_init(matrix, dim, seed=None):
    np_random = numpy.random.RandomState(seed)
    w = np_random.uniform(size=(matrix.shape[0], dim))
    h = np_random.uniform(size=(dim, matrix.shape[1]))
    return w, h

def _improve_beta_divergence(orig, current, w, h, epsilon=1e-9, beta=2.0):
    if beta < 1:
        phi = 1.0/(2.0-beta)
    elif beta <= 2.0:
        phi = 1.0
    else:
        phi = 1.0/(beta - 1.0)

    wt = w.transpose()
    h *= (wt.dot(orig * current**(beta - 2))/(wt.dot(current**(beta - 1)) + epsilon))**phi
    ht = h.transpose()
    current = w.dot(h)
    w *= ((orig * current**(beta - 2)).dot(ht)/((current**(beta - 1)).dot(ht) + epsilon))**phi

    return w.dot(h), w, h


def _improve_euclidean_distance(orig, current, w, h, epsilon=1e-9):
    wt = w.transpose()
    h *= wt.dot(orig)/(wt.dot(current) + epsilon)
    ht = h.transpose()
    current = w.dot(h)
    w *= orig.dot(ht)/(current.dot(ht) + epsilon)
    return w.dot(h), w, h

def _improve_kl_diveregence(orig, current, w, h, epsilon=1e-9):
    ws = w.sum(axis=0)
    wt = (w/(ws + epsilon)).transpose()
    h *= wt.dot(orig/(current + epsilon))
    ht = h.transpose()
    hs = ht.sum(axis=0)
    current = w.dot(h)
    w *= (orig/(current + epsilon)).dot(ht/(hs + epsilon))
    return w.dot(h), w, h

def nmf(matrix,
        dim=None,
        distance="euclid",
        init=svd_init,
        max_iter=10000,
        threshould=0.001,
        epsilon=1e-9,
        seed=None):
    """Non-negative Matrix Factorization function

    :param numpy.array matrix: Matrix to decompose
    :param int dim: dimension of matrix
    :param float distance: distance to minimize. choose "euclid" or "kl".
                       euclid: Euclid distance
                       k: Kullback Leibler divergence
                     default: "euclid"
    :param int max_iter: max #iteration of calculation
                     defau:t] 10000
    :param float thresould: threshould to regard as converged
    :param float epsilon: epsilon to avoid zero division
    :param int seed: random seed

    :return: factorized matrix w and h
    """
    max_rank = min(matrix.shape)
    dim = min(dim, max_rank) if dim is not None else max_rank

    if distance == "euclid":
        _improve = _improve_euclidean_distance
    elif distance == "kl":
        _improve = _improve_kl_diveregence
    elif distance == "beta":
        _improve = _improve_beta_divergence

    w, h = init(matrix, dim, seed)

    wh = w.dot(h)
    prev_norm = numpy.linalg.norm(matrix - wh)
    for _ in range(max_iter):
        wh, w, h = _improve(matrix, wh, w, h, epsilon)

        norm = numpy.linalg.norm(matrix - wh)
        improvement = (prev_norm - norm)/prev_norm
        if improvement < threshould:
            break
        prev_norm = norm

    return w, h
