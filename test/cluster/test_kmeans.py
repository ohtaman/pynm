# -*- coding:utf-8 -*-

import logging

from nose.tools import *
import numpy

from pynm.cluster import kmeans


logger = logging.getLogger(__name__)


@istest
def can_treat_2dim_dataset():
    dataset = [[0, 1],
               [0, 3],
               [-1, -3],
               [-3, -2],
               [-2, -4]]

    centers, clusters, converge = kmeans.kmeans(dataset, 2)
    eq_(2, len(centers))
    logger.info(centers)
    ok_((centers[0] == (-2, -3)).all()
        or (centers[0] == (0, 2)).all(), "Failed to create right cluster")
