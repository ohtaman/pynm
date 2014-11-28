# -*- coding:utf-8 -*-

import numpy
import numpy.random
from numpy.linalg import norm

def nmf(matrix, d=None, max_iter=10000, threshould=0.0001):
    if d is None:
        d = min(matrix.shape)

    w = numpy.random.random(size=(matrix.shape[0], d))
    h = numpy.random.random(size=(d, matrix.shape[1]))

    wh = w.dot(h)
    n1 = norm(matrix - wh)
    for _ in range(max_iter):
        wt = w.transpose()
        h *= wt.dot(matrix)/wt.dot(w).dot(h)
        ht = h.transpose()
        w *= matrix.dot(ht)/w.dot(h).dot(ht)
        wh = w.dot(h)
        n2 = norm(matrix - wh)
        if True:
            print ("n: %s, n2: %s, (n - n2)/n: %s" % (n1, n2, (n1 - n2)/n1))
        if (n1 - n2)/n1 < threshould:
            break
        n1 = n2

    return w, h
