# -*- coding: utf-8 -*-

import unittest
import os
import sys
import pathlib
import unittest.mock as mock

from unittest.mock import patch

import numpy as np


class TestWindowArray(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_window_access(self):
        from wxbtool.data.dataset import WindowArray
        w = WindowArray(np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=0, step=2)
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(0, np.array(w[0, 0, 0], dtype=np.float32))
        self.assertEqual(1, np.array(w[0, 0, 1], dtype=np.float32))
        self.assertEqual(2, np.array(w[0, 0, 2], dtype=np.float32))
        self.assertEqual(3, np.array(w[0, 1, 0], dtype=np.float32))
        self.assertEqual(4, np.array(w[0, 1, 1], dtype=np.float32))
        self.assertEqual(5, np.array(w[0, 1, 2], dtype=np.float32))

        w = WindowArray(np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=1, step=2)
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(6, np.array(w[0, 0, 0], dtype=np.float32))
        self.assertEqual(7, np.array(w[0, 0, 1], dtype=np.float32))
        self.assertEqual(8, np.array(w[0, 0, 2], dtype=np.float32))
        self.assertEqual(9, np.array(w[0, 1, 0], dtype=np.float32))
        self.assertEqual(10, np.array(w[0, 1, 1], dtype=np.float32))
        self.assertEqual(11, np.array(w[0, 1, 2], dtype=np.float32))

        w = WindowArray(np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=0, step=2)
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(12, np.array(w[1, 0, 0], dtype=np.float32))
        self.assertEqual(13, np.array(w[1, 0, 1], dtype=np.float32))
        self.assertEqual(14, np.array(w[1, 0, 2], dtype=np.float32))
        self.assertEqual(15, np.array(w[1, 1, 0], dtype=np.float32))
        self.assertEqual(16, np.array(w[1, 1, 1], dtype=np.float32))
        self.assertEqual(17, np.array(w[1, 1, 2], dtype=np.float32))

        w = WindowArray(np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3), shift=1, step=2)
        self.assertEqual((2, 2, 3), w.shape)
        self.assertEqual((2, 3), w[0].shape)
        self.assertEqual(18, np.array(w[1, 0, 0], dtype=np.float32))
        self.assertEqual(19, np.array(w[1, 0, 1], dtype=np.float32))
        self.assertEqual(20, np.array(w[1, 0, 2], dtype=np.float32))
        self.assertEqual(21, np.array(w[1, 1, 0], dtype=np.float32))
        self.assertEqual(22, np.array(w[1, 1, 1], dtype=np.float32))
        self.assertEqual(23, np.array(w[1, 1, 2], dtype=np.float32))

