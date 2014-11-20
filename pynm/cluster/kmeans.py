# -*- coding:utf-8 -*-

import random

import numpy
import numpy.linalg

def kmeans_pp(dataset, k):
    def get_weight(centers, data):
        return min(numpy.linalg.norm(center - data) for center in centers)

    centers = [random.choice(dataset)]
    for _ in range(k - 1):
        centers.append(
            numpy.random.choice(
                dataset,
                p=(get_weight(data) for data in dataset)
            )
        )
    return centers

def kmeans(dataset, k, init=kmeans_pp, max_iter=1024):
    centers = initialize(dataset, k)
    clusters = []
    for _ in range(max_iter):
        converge = True
        for i, d in enumerate(data):
            cluster = min(enumerate(centers), lambda x: numpy.linalg.norm(d - x[1]))[0]
            if clusters[i] != cluster:
                converge = False
                clusters[i] = cluster
        if converge:
            break
    return centers, clusters, converge
