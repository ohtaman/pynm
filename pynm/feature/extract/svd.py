# -*- coding:utf-8 -*-

import numpy.random
import numpy.linalg


def _orthogonalize(matrix):
    return numpy.linalg.qr(matrix)[0]


def svd(matrix, dim=None, k=10, u=True, s=True, v=True, seed=None, approx=True):
    """Singular Value Decomposion function

    :param numpy.array matrix: Matrix to decompose
    :param int dim: dimension of matrix
                    default: min(matrix.shape)
    :param int k: additional dimension to improve accuracy of approximation.
                  default: 10
    :param bool u: calculate U matrix
                   defaut: True
    :param bool s: calculate S vector
                   defaut: True
    :param bool v: calculate V matrix
                   defaut: True
    :param int seed: random seed
    :param bool approx: use approximation. (see http://arxiv.org/abs/0909.4061)

    :return: factorized matrix U, S and V
    """

    max_rank = min(matrix.shape)
    dim = dim if dim is not None else max_rank
    # to reduce error, calc with dim=internal_dim internally
    internal_dim = min(dim + k, max_rank)

    if internal_dim == max_rank or not approx:
        u_, s_, v_ = numpy.linalg.svd(matrix)
        return u_[:, :dim], s_[:dim], v_[:dim, :]

    # cost effective
    if matrix.shape[0] > matrix.shape[1]:
        v_, s_, u_ = svd(matrix.transpose(), dim, k, v, s, u, seed, approx)
        return (u_.transpose() if u else None,
                s_ if s else None,
                v_.transpose() if v else None)

    np_random = numpy.random.RandomState(seed)
    o = np_random.normal(size=(matrix.shape[0], internal_dim))
    y = _orthogonalize(matrix.transpose().dot(o))
    b = matrix.dot(y)
    p = np_random.normal(size=(internal_dim, internal_dim))
    z = _orthogonalize(b.dot(p))
    c = z.transpose().dot(b)
    u_, s_, v_ = numpy.linalg.svd(c)
    return (z.dot(u_[:, :dim]) if u else None,
            s_[:dim] if s else None,
            v_[:dim, :].dot(y.transpose()) if v else None)
