# -*- coding:utf-8 -*-

import os
import shutil
import sys
import tempfile
if sys.version_info[0] == 2:
    import mock
else:
    from unittest import mock

from nose.tools import *

from pynm.commands import bandit

workdir = tempfile.mkdtemp()
dirname = os.path.dirname(__file__)
testee = None

def setup():
    global testee
    testee = bandit.BanditCommand()


def teardown():
    shutil.rmtree(workdir)


def test_cli_can_run():
    input_file = os.path.join(workdir, "input.json")
    output_file = os.path.join(workdir, "output.json")
    with open(input_file, 'w') as i_:
        i_.write('{"type": "choose", "arms": [1,2,3,4,5], "features": {"a": 1}}')

    args = mock.Mock()
    args.algorithm = 'ucb1'
    args.input = input_file
    args.output = output_file
    testee.run(args)

    with open(output_file) as o_:
        eq_(1, int(o_.read()))
