# -*- coding: utf-8 -*-

import os
import os.path as path
import hashlib
import requests
import logging
import json

import xarray as xr
import numpy as np

import msgpack
import msgpack_numpy as m
m.patch()


from itertools import product
from torch.utils.data import Dataset

from wxbtool.data.variables import codes, vars2d, vars3d

logger = logging.getLogger()

all_levels = ['50', '100', '150', '200', '250', '300', '400', '500', '600', '700', '850', '925', '1000']


class WxDataset(Dataset):
    def __init__(self, root, resolution, years, vars, levels, step=1, input_span=2, pred_shift=24, pred_span=1):
        self.root = root
        self.resolution = resolution
        self.input_span = input_span
        self.step = step
        self.pred_shift = pred_shift
        self.pred_span = pred_span
        self.years = years
        self.vars = vars
        self.levels = levels
        self.inputs = {}
        self.targets = {}
        self.shapes = {
            'inputs': {},
            'targets': {},
        }

        if resolution == '5.625deg':
            self.height = 13
            self.width = 32
            self.length = 64

        code = '%s:%s:%s:%s:%s:%s:%s:%s' % (resolution, years, vars, levels, step, input_span, pred_shift, pred_span)
        hashstr = hashlib.md5(code.encode('utf-8')).hexdigest()
        self.hashcode = hashstr

        dumpdir = path.abspath('%s/.cache/%s' % (self.root, hashstr))
        if not path.exists(dumpdir):
            os.makedirs(dumpdir)
            self.load(dumpdir)

        self.memmap(dumpdir)

    def load(self, dumpdir):
        levels_selector = []
        for l in self.levels:
            levels_selector.append(all_levels.index(l))
        selector = np.array(levels_selector, dtype=np.int)

        lastvar, input, target = None, None, None
        size, dti, dto = self.init_holders(self.vars)
        for var, yr in product(self.vars, self.years):
            if var in vars3d:
                length, chi, cho = self.load_3ddata(yr, var, selector)
            elif var in vars2d:
                length, chi, cho = self.load_2ddata(yr, var)
            else:
                raise ValueError('variable %s dose not supported!' % var)
            size = size + length
            dti[var].append(chi)
            dto[var].append(cho)

            if lastvar and lastvar != var:
                input = np.concatenate(tuple(dti[lastvar]), axis=0)
                target = np.concatenate(tuple(dto[lastvar]), axis=0)
                dti[lastvar] = None
                dto[lastvar] = None

                self.inputs[lastvar] = input
                self.targets[lastvar] = target
                self.dump_var(dumpdir, lastvar)

            lastvar = var

        # dump the last
        self.inputs[lastvar] = input
        self.targets[lastvar] = target
        self.dump_var(dumpdir, lastvar)

        with open('%s/shapes.json' % dumpdir, mode='w') as fp:
            json.dump(self.shapes, fp)

        self.size = size // len(self.vars)
        logger.info('total %s items loaded!', self.size)

    def init_holders(self, vars):
        return 0, {k: [] for k in vars}, {k: [] for k in vars}

    def load_2ddata(self, year, var):
        data_path = '%s/%s/%s_%d_%s.nc' % (self.root, var, var, year, self.resolution)
        ds = xr.open_dataset(data_path)
        ds = ds.transpose('time', 'lat', 'lon')
        dt = np.array(ds[codes[var]].data, dtype=np.float32)
        logger.info('%s[%d]: %s', var, year, str(dt.shape))

        length = 365 * 24 - (self.input_span * self.step + self.pred_span * self.step + self.pred_shift)
        dti, dto = (
            np.zeros([length, self.input_span, self.width, self.length], dtype=np.float32),
            np.zeros([length, self.pred_span, self.width, self.length], dtype=np.float32)
        )

        for ix in range(0, length):
            pt = ix
            for jx in range(self.input_span):
                dti[pt, jx, :, :] = dt[ix + jx * self.step, :, :]
            for kx in range(self.pred_span):
                dto[pt, kx, :, :] = dt[ix + self.pred_shift + kx * self.step, :, :]

        ds.close()

        return length, dti, dto

    def load_3ddata(self, year, var, selector):
        data_path = '%s/%s/%s_%d_%s.nc' % (self.root, var, var, year, self.resolution)
        ds = xr.open_dataset(data_path)
        ds = ds.transpose('time', 'level', 'lat', 'lon')
        dt = np.array(ds[codes[var]].data, dtype=np.float32)
        logger.info('%s[%d]: %s', var, year, str(dt.shape))

        height = selector.shape[0]

        length = 365 * 24 - (self.input_span * self.step + self.pred_span * self.step + self.pred_shift)
        dti, dto = (
            np.zeros([length, self.input_span, height, self.width, self.length], dtype=np.float32),
            np.zeros([length, self.pred_span, height, self.width, self.length], dtype=np.float32)
        )

        for ix in range(0, length):
            pt = ix
            for jx in range(self.input_span):
                dti[pt, jx, :, :, :] = dt[ix + jx * self.step, selector, :, :]
            for kx in range(self.pred_span):
                dto[pt, kx, :, :, :] = dt[ix + self.pred_shift + kx * self.step, selector, :, :]

        ds.close()

        return length, dti, dto

    def dump_var(self, dumpdir, var):
        input_dump = '%s/input_%s.npy' % (dumpdir, var)
        target_dump = '%s/target_%s.npy' % (dumpdir, var)

        self.shapes['inputs'][var] = self.inputs[var].shape
        np.save(input_dump, self.inputs[var])
        del self.inputs[var]

        self.shapes['targets'][var] = self.targets[var].shape
        np.save(target_dump, self.targets[var])
        del self.targets[var]

    def memmap(self, dumpdir):
        with open('%s/shapes.json' % dumpdir) as fp:
            shapes = json.load(fp)

        for var in self.vars:
            input_dump = '%s/input_%s.npy' % (dumpdir, var)
            target_dump = '%s/target_%s.npy' % (dumpdir, var)

            shape = shapes['inputs'][var]
            total_size = np.prod(shape)
            input = np.memmap(input_dump, dtype=np.float32, mode='r')
            shift = input.shape[0] - total_size
            self.inputs[var] = np.reshape(input[shift:], shape)

            shape = shapes['targets'][var]
            total_size = np.prod(shape)
            target = np.memmap(target_dump, dtype=np.float32, mode='r')
            shift = target.shape[0] - total_size
            self.targets[var] = np.reshape(target[shift:], shape)

    def __len__(self):
        return self.inputs[self.vars[0]].shape[0]

    def __getitem__(self, item):
        inputs, targets = {}, {}
        for var in self.vars:
            inputs.update({var: self.inputs[var][item:item+1]})
            targets.update({var: self.targets[var][item:item+1]})
        return inputs, targets


class WxDatasetClient(Dataset):
    def __init__(self, url, phase, resolution, years, vars, levels, step=1, input_span=2, pred_shift=24, pred_span=1):
        self.url = url
        self.phase = phase

        code = '%s:%s:%s:%s:%s:%s:%s:%s' % (resolution, years, vars, levels, step, input_span, pred_shift, pred_span)
        self.hashcode = hashlib.md5(code.encode('utf-8')).hexdigest()

    def __len__(self):
        url = '%s/%s/%s'% (self.url, self.hashcode, self.phase)
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception('http error %s: %s' % (r.status_code, r.text))

        data = msgpack.loads(r.content)

        return data['size']

    def __getitem__(self, item):
        url = '%s/%s/%s/%d'% (self.url, self.hashcode, self.phase, item)
        r = requests.get(url)
        if r.status_code != 200:
            raise Exception('http error %s: %s' % (r.status_code, r.text))

        data = msgpack.loads(r.content)
        for key, val in data.items():
            for var, blk in val.items():
                val[var] = np.copy(blk)

        return data['inputs'], data['targets']
