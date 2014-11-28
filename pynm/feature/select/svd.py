# -*- coding:utf-8 -*-

import numpy.random
import numpy.linalg

def _orthogonalize(matrix):
    return numpy.linalg.qr(matrix)[0]

def svd(matrix, d=None, k=10, u=True, s=True, v=True, seed=None, approx=True):

    max_rank = min(matrix.shape)
    d = d if d is not None else max_rank
    # to reduce error, calc with dim=internal_d internally
    internal_d = min(d + k, max_rank)

    if internal_d == max_rank or not approx:
        u_, s_, v_ = numpy.linalg.svd(matrix)
        return u_[:, :d], s_[:d], v_[:d, :]

    # cost effective
    if matrix.shape[0] > matrix.shape[1]:
        v_, s_, u_ = svd(matrix.transpose(), d, k, v, s, u, seed, approx)
        return (u_.transpose() if u else None,
                s_ if s else None,
                v_.transpose() if v else None)

    np_random = numpy.random.RandomState(seed)
    o = np_random.normal(size=(matrix.shape[0], internal_d))
    y = _orthogonalize(matrix.transpose().dot(o))
    b = matrix.dot(y)
    p = np_random.normal(size=(internal_d, internal_d))
    z = _orthogonalize(b.dot(p))
    c = z.transpose().dot(b)
    u_, s_, v_ = numpy.linalg.svd(c)
    return (z.dot(u_[:, :d]) if u else None,
            s_[:d] if s else None,
            v_[:d, :].dot(y.transpose()) if v else None)
