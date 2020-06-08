# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import logging

from itertools import product
from torch.utils.data import Dataset

from wxbtool.variables import codes
from wxbtool.normalization import normalizors

logger = logging.getLogger()


class WxDataset(Dataset):
    def __init__(self, root, resolution, years, vars, input_span=2, pred_shift=24):
        self.root = root
        self.resolution = resolution
        self.input_span = input_span
        self.pred_shift = pred_shift

        size, dti, dto = self.init_holders(vars)
        for var, yr in product(vars, years):
            length, chi, cho = self.load_3ddata(yr, var)
            size = size + length
            dti[var].append(chi)
            dto[var].append(cho)

        self.size = size // len(vars)
        self.dti = np.concatenate(tuple((np.concatenate(tuple(dti[v]), axis=0) for v in vars)), axis=1)
        self.dto = np.concatenate(tuple((np.concatenate(tuple(dto[v]), axis=0) for v in vars)), axis=1)
        logger.info('total %s items loaded!', self.size)

        if resolution == '5.625deg':
            self.levels = 11
            self.width = 32
            self.length = 64

    def init_holders(self, vars):
        return 0, {k: [] for k in vars}, {k: [] for k in vars}

    def load_3ddata(self, year, var):
        data_path = '%s/%s/%s_%d_%s.nc' % (self.root, var, var, year, self.resolution)
        ds = xr.open_dataset(data_path)
        ds = ds.transpose('time', 'level', 'lat', 'lon')
        dt = np.array(ds[codes[var]].data, dtype=np.float32)
        logger.info('%s[%d]: %s', var, year, str(dt.shape))

        length = 365 * 24 - (self.input_span + self.pred_shift)
        dti, dto = (
            np.zeros([length // self.input_span, self.input_span, self.levels, self.width, self.length], dtype=np.float32),
            np.zeros([length // self.input_span, 1, self.input_span, self.levels, self.width, self.length], dtype=np.float32)
        )

        for ix in range(0, length, self.input_span):
            pt = ix // self.input_span
            dto[pt, 0, :, :, :] = dt[self.pred_shift + self.input_span - 1 + ix, :, :, :]
            for jx in range(self.input_span):
                dti[pt, jx, :, :, :] = dt[ix + jx, :, :, :]

        return length // self.input_span, normalizors[var](dti), normalizors[var](dto)

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        return {'input3d': self.dti[item:item+1], 'target3d': self.dto[item:item+1]}
