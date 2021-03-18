# -*- coding: utf-8 -*-

import numpy as np
import wxbtool.data.constants as cnst


class Evaluator:
    def __init__(self, resolution, root):
        self.root = root
        self.resolution = resolution

        if resolution == '5.625deg':
            self.levels = 11
            self.width = 32
            self.length = 64

        self.weight = cnst.load_area_weight(resolution, root)

    def weighted_rmse(self, ground_truth, forcast):
        err = forcast - ground_truth
        return np.sqrt(np.mean(self.weight * err * err) + 1e-10)
