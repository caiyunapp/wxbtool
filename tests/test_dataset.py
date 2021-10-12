# -*- coding: utf-8 -*-

import unittest
import os
import sys
import pathlib
import unittest.mock as mock

from unittest.mock import patch


class TestTest(unittest.TestCase):

    def setUp(self):
        for g in (pathlib.Path(__file__).parent.absolute() / '.cache').glob('*/*'):
            g.unlink()
        for g in (pathlib.Path(__file__).parent.absolute() / '.cache').glob('*'):
            g.rmdir()
        if (pathlib.Path(__file__).parent.absolute() / '.cache').exists():
            (pathlib.Path(__file__).parent.absolute() / '.cache').rmdir()

    def tearDown(self):
        pass

    @mock.patch.dict(os.environ, {"WXBHOME": str(pathlib.Path(__file__).parent.absolute())})
    def test_dataset(self):
        import wxbtool.wxb as wxb

        testargs = ['wxb', 'dserve', '-m', 'models.modeltest', '-s', 'Setting3d', '-t', 'true']
        with patch.object(sys, 'argv', testargs):
            wxb.main()

        testargs = ['wxb', 'test', '-m', 'models.modeltest', '-b', '1']
        with patch.object(sys, 'argv', testargs):
            wxb.main()
