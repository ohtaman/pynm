# -*- coding:utf-8 -*-

from numpy import random, linalg


def orth(matrix):
    return linalg.qr(matrix)[0]

def random_orthogonal_matrix(shape):
    return linalg.qr(random.normal(size=shape))[0]

def svd(matrix, d=None, k=10, u=True, s=True, v=True):
    if d is None:
        d = min(matrix.shape)

    internal_d = min(d + k, min(matrix.shape))
    if internal_d == min(matrix.shape):
        u_, s_, v_ = linalg.svd(matrix)
        return u_[:, :d], s_[:d], v_[:, :d]

    if matrix.shape[0] > matrix.shape[1]:
        u_, s_, v_ = svd(matrix.transpose(), d, k, u, s, v)
        return v_.transpose(), s_, u_.transpose()

    o = random.normal(size=(matrix.shape[0], internal_d))
    y = orth(matrix.transpose().dot(o))
    b = matrix.dot(y)
    p = random.normal(size=(internal_d, internal_d))
    z = orth(b.dot(p))
    c = z.transpose().dot(b)
    u_, s_, v_ = linalg.svd(c)
    return (z.dot(u_[:, :d]) if u else None,
            s_[:d] if s else None,
            y.dot(v_[:, :d]) if v else None)
