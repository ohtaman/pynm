# -*- coding:utf-8 -*-

import os

from nose.tools import *

from pynm.commands import metric

dirname = os.path.dirname(__file__)
testee = None

def setup():
    global testee
    testee = metric.ItmlCommand()


def teardown():
    pass


def test_cli_load_data():
    input_data_file = os.sep.join((dirname,
                                   '..',
                                   'feature',
                                   'metric',
                                   'input_data.csv'))
    with open(input_data_file) as input_data:
        header, data = testee.load_data(input_data,
                                        delimiter='\t',
                                        has_header=False)
    eq_(data[10][0], 0.0)
    eq_(data[10][1], 0.0)
    eq_(data[10][2], 1.0)
    eq_(data[10][3], 1.0)
    eq_(header, None)


def test_cli_load_labels():
    input_labels_file = os.sep.join((dirname,
                                     '..',
                                     'feature',
                                     'metric',
                                     'input_labels.csv'))
    with open(input_labels_file) as input_labels:
        labels = testee.load_labels(input_labels)
    eq_(labels, [1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1])


def test_cli_load_pairs():
    input_pairs_file = os.sep.join((dirname,
                                    '..',
                                    'feature',
                                    'metric',
                                    'input_pairs.csv'))
    with open(input_pairs_file) as input_pairs:
        pairs = testee.load_pairs(input_pairs)
    eq_(pairs[0], (0, 1, True))
    eq_(pairs[1], (0, 2, False))
    eq_(pairs[2], (0, 3, False))
    eq_(pairs[3], (0, 4, True))
