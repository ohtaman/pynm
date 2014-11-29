# -*- coding:utf-8 -*-

import numpy.random
import numpy.linalg


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
        max_iter=10000,
        threshould=0.001,
        epsilon=1e-9,
        seed=None):
    max_rank = min(matrix.shape)
    dim = min(dim, max_rank) if dim is not None else max_rank

    if distance == "euclid":
        _improve = _improve_euclidean_distance
    elif distance == "kl":
        _improve = _improve_kl_diveregence

    np_random = numpy.random.RandomState(seed)
    w = np_random.uniform(size=(matrix.shape[0], dim))
    h = np_random.uniform(size=(dim, matrix.shape[1]))

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
