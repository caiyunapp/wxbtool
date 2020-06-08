# -*- coding: utf-8 -*-

import xarray as xr
import numpy as np
import logging

from itertools import product
from torch.utils.data import Dataset

from wxbtool.variables import codes, vars2d, vars3d
from wxbtool.normalization import normalizors

logger = logging.getLogger()


class WxDataset(Dataset):
    def __init__(self, root, resolution, years, vars, input_span=2, pred_shift=24):
        self.root = root
        self.resolution = resolution
        self.input_span = input_span
        self.pred_shift = pred_shift
        self.vars = vars
        self.inputs = {}
        self.targets = {}

        if resolution == '5.625deg':
            self.levels = 11
            self.width = 32
            self.length = 64

        size, dti, dto = self.init_holders(vars)
        for var, yr in product(vars, years):
            if var in vars3d:
                length, chi, cho = self.load_3ddata(yr, var)
            elif var in vars2d:
                length, chi, cho = self.load_2ddata(yr, var)
            else:
                raise ValueError('variable %s dose not supported!' % var)
            size = size + length
            dti[var].append(chi)
            dto[var].append(cho)

        self.size = size // len(vars)
        self.inputs.update({v: np.concatenate(tuple(dti[v]), axis=0) for v in vars})
        self.targets.update({v: np.concatenate(tuple(dto[v]), axis=0) for v in vars})
        logger.info('total %s items loaded!', self.size)

    def init_holders(self, vars):
        return 0, {k: [] for k in vars}, {k: [] for k in vars}

    def load_2ddata(self, year, var):
        data_path = '%s/%s/%s_%d_%s.nc' % (self.root, var, var, year, self.resolution)
        ds = xr.open_dataset(data_path)
        ds = ds.transpose('time', 'lat', 'lon')
        dt = np.array(ds[codes[var]].data, dtype=np.float32)
        logger.info('%s[%d]: %s', var, year, str(dt.shape))

        length = 365 * 24 - (self.input_span + self.pred_shift)
        dti, dto = (
            np.zeros([length // self.input_span, self.input_span, self.width, self.length], dtype=np.float32),
            np.zeros([length // self.input_span, 1, self.width, self.length], dtype=np.float32)
        )

        for ix in range(0, length, self.input_span):
            pt = ix // self.input_span
            dto[pt, 0, :, :] = dt[self.pred_shift + self.input_span - 1 + ix, :, :]
            for jx in range(self.input_span):
                dti[pt, jx, :, :] = dt[ix + jx, :, :]

        return length // self.input_span, normalizors[var](dti), normalizors[var](dto)

    def load_3ddata(self, year, var):
        data_path = '%s/%s/%s_%d_%s.nc' % (self.root, var, var, year, self.resolution)
        ds = xr.open_dataset(data_path)
        ds = ds.transpose('time', 'level', 'lat', 'lon')
        dt = np.array(ds[codes[var]].data, dtype=np.float32)
        logger.info('%s[%d]: %s', var, year, str(dt.shape))

        length = 365 * 24 - (self.input_span + self.pred_shift)
        dti, dto = (
            np.zeros([length // self.input_span, self.input_span, self.levels, self.width, self.length], dtype=np.float32),
            np.zeros([length // self.input_span, 1, self.levels, self.width, self.length], dtype=np.float32)
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
        inputs, targets = {}, {}
        for var in self.vars:
            inputs.update({var: self.inputs[var][item:item+1]})
            targets.update({var: self.targets[var][item:item+1]})
        return inputs, targets
