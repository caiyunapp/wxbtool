# -*- coding: utf-8 -*-

import random

import numpy as np
import torch as th
import torch.nn as nn

from wxbtool.constants import load_area_weight, load_lsm, load_slt, load_orography, load_lat2d, load_lon2d
from wxbtool.evaluation import Evaluator
from wxbtool.dataset import WxDataset


def cast(element):
    element = np.array(element, dtype=np.float32)
    tensor = th.FloatTensor(element)
    return tensor


class Meta2d:
    def __init__(self):
        resolution = '5.625deg'
        root = 'weatherbench/5.625deg/'
        eva = Evaluator(resolution, root)

        name = 'base'

        step = 4
        input_span = 3
        pred_shift = 72

        levels = ['300', '500', '700', '850', '1000']
        height = len(levels)

        vars = ['geopotential', 'toa_incident_solar_radiation', '2m_temperature', 'temperature', 'total_cloud_cover']
        vars_in = ['z500', 'z1000', 'tau', 't850', 'tcc', 't2m', 'tisr']
        vars_out = ['t850']

        years_train = [
            1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989,
            1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
            2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
            2010, 2011, 2012, 2013
        ]
        years_test = [2014]
        years_eval = [2014, 2015, 2016, 2017, 2018]

        dataset_train, dataset_test, dataset_eval = None, None, None
        train_size = -1
        test_size = -1
        eval_size = -1

    def load_dataset(mode='train'):
        global dataset_train, dataset_test, dataset_eval, train_size, test_size, eval_size

        if mode == 'train':
            dataset_train, dataset_eval = (
                    WxDataset(mdls.root, mdls.resolution, years_train, vars, levels, input_span=input_span, pred_shift=pred_shift, pred_span=1, step=step),
                    WxDataset(mdls.root, mdls.resolution, years_eval, vars, levels, input_span=input_span, pred_shift=pred_shift, pred_span=1, step=step)
            )
            train_size = dataset_train.size
            eval_size = dataset_eval.size
        else:
            dataset_test = WxDataset(mdls.root, mdls.resolution, years_test, vars, input_span=input_span, pred_shift=pred_shift, pred_span=1, step=step)
            test_size = dataset_test.size


class Model2d(nn.Module):
    def __init__(self):
        super().__init__()
        lsm = cast(load_lsm(resolution, root))
        slt = cast(load_slt(resolution, root))
        oro = cast(load_orography(resolution, root))
        aw = cast(load_area_weight(resolution, root))

        phi = cast(load_lat2d(resolution, root)) * np.pi / 180
        theta = cast(load_lon2d(resolution, root)) * np.pi / 180
        x, y = np.meshgrid(np.linspace(0, 1, num=32), np.linspace(0, 1, num=64))
        x = cast(x)
        y = cast(y)

        lsm.requires_grad = False
        slt.requires_grad = False
        oro.requires_grad = False
        aw.requires_grad = False
        phi.requires_grad = False
        theta.requires_grad = False
        x.requires_grad = False
        y.requires_grad = False

        dt = th.cos(phi)
        self.weight = dt / dt.mean()
        print(th.sum(self.weight).item())
        print(th.sum(th.relu(self.weight)).item())

        lsm = ((lsm - 0.33707827) / 0.45900375).view(1, 1, 32, 64)
        slt = ((slt - 0.67920434) / 1.1688842).view(1, 1, 32, 64)
        oro = ((oro - 379.4976) / 859.87225).view(1, 1, 32, 64)
        self.constant = th.cat((lsm, slt, oro), dim=1)
        self.constant_size = self.constant.size()[1]
        self.phi = phi
        self.theta = theta
        self.x = x
        self.y = y

        self.constant_cache = {}
        self.weight_cache = {}
        self.phi_cache = {}
        self.theta_cache = {}
        self.x_cache = {}
        self.y_cache = {}

        self.grid_equi = th.zeros(1, 48, 48, 2)
        self.grid_polr = th.zeros(1, 48, 48, 2)
        self.grid_equi_cache = {}
        self.grid_polr_cache = {}

    def get_constant(self, input, device):
        if device not in self.constant_cache:
            self.constant_cache[device] = self.constant.to(device)
        return self.constant_cache[device]

    def get_weight(self, device):
        if device not in self.weight_cache:
            self.weight_cache[device] = self.weight.to(device)
        return self.weight_cache[device]

    def get_phi(self, device):
        if device not in self.phi_cache:
            self.phi_cache[device] = self.phi.to(device)
        return self.phi_cache[device]

    def get_theta(self, device):
        if device not in self.theta_cache:
            self.theta_cache[device] = self.theta.to(device)
        return self.theta_cache[device]

    def get_x(self, device):
        if device not in self.x_cache:
            self.x_cache[device] = self.x.to(device)
        return self.x_cache[device]

    def get_y(self, device):
        if device not in self.y_cache:
            self.y_cache[device] = self.y.to(device)
        return self.y_cache[device]

    def get_grid_equi(self, device):
        if device not in self.grid_equi_cache:
            self.grid_equi_cache[device] = self.grid_equi.to(device)
        return self.grid_equi_cache[device]

    def get_grid_polr(self, device):
        if device not in self.grid_polr_cache:
            self.grid_polr_cache[device] = self.grid_polr.to(device)
        return self.grid_polr_cache[device]


class Base2d(Model2d):
    def __init__(self, enable_da=False):
        super().__init__()
        self.enable_da = enable_da

    def update_da_status(self, batch):
        if self.enable_da and self.training:
            self.lng_shift = []
            self.flip_status = []
            for _ in range(batch):
                self.lng_shift.append(random.randint(0, 64))
                self.flip_status.append(random.randint(0, 1))

    def augment_data(self, data):
        if self.enable_da and self.training:
            augmented = []
            b, c, w, h = data.size()
            for _ in range(b):
                slice = data[_:_+1]
                shift = self.lng_shift[_]
                flip = self.flip_status[_]
                slice = slice.roll(shift, dims=(3,))
                if flip == 1:
                    slice = th.flip(slice, dims=(2, 3))
                augmented.append(slice)
            data = th.cat(augmented, dim=0)
        return data

    def get_augmented_constant(self, input):
        constant = self.get_constant(input, input.device).repeat(input.size()[0], 1, 1, 1)
        constant = self.augment_data(constant)
        phi = self.get_phi(input.device).repeat(input.size()[0], 1, 1, 1)
        theta = self.get_theta(input.device).repeat(input.size()[0], 1, 1, 1)
        constant = th.cat((constant, phi, theta), dim=1)
        return constant

    def get_inputs(self, **kwargs):
        return {}, None

    def get_targets(self, **kwargs):
        return {}, None

    def get_results(self, **kwargs):
        return {}, None

    def forward(self, *args, **kwargs):
        return {}

    def lossfun(self, inputs, result, target):
        return 0.0
