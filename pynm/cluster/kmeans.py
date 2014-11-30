# -*- coding:utf-8 -*-

import logging
import random

import numpy
import numpy.linalg

logger = logging.getLogger(__name__)

def kmeans_pp(dataset, k, seed=None):
    def get_weight(centers, data):
        return min(numpy.linalg.norm(center - data) for center in centers)**2

    def _choice(centers, dataset):
        weights = list(get_weight(centers, data) for data in dataset)
        sum_weights = int(sum(weights))
        rand = random.randint(1, sum_weights)
        tot = 0
        for i in range(len(weights)):
            tot += weights[i]
            if rand < tot:
                return dataset[i]

    np_random = numpy.random.RandomState(seed)
    centers = [numpy.array(random.choice(dataset)),]
    for _ in range(k - 1):
        centers.append(numpy.array(_choice(centers, dataset)))
    return centers

def get_centers(clusters):
    return list(numpy.average(cluster, axis=0) for cluster in clusters)

def kmeans(dataset,
           k,
           norm=numpy.linalg.norm,
           init=kmeans_pp,
           max_iter=1024,
           seed=None):
    """K-Means clustering function

    :param dataset: dataset to aply K-Means clustering
    :param int k: #clusters
    :param callable norm: norm function
    :param string init: initialization algorithm
                        "kmeans++" only
    :param int max_iter: max number of iteration
                         default: 1024
    :param int seed: random seed

    :return: centers, clusters, converge or not
    """
    centers = init(dataset, k, seed)
    cluster_ids = [0]*len(dataset)

    for _ in range(max_iter):
        converge = True
        clusters = list([] for _ in range(k))

        for i, data in enumerate(dataset):
            cluster_id = numpy.argmin(list(numpy.linalg.norm(center - data) for center in centers))
            if cluster_ids[i] != cluster_id:
                converge = False
                cluster_ids[i] = cluster_id
            clusters[cluster_id].append(numpy.array(data))
        if converge:
            break
        centers = get_centers(clusters)

    return get_centers(clusters), clusters, converge
