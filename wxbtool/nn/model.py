# -*- coding: utf-8 -*-

import random

import numpy as np
import torch as th
import torch.nn as nn

from wxbtool.data.constants import load_area_weight, load_lsm, load_slt, load_orography, load_lat2d, load_lon2d
from wxbtool.util.evaluation import Evaluator
from wxbtool.data.dataset import WxDataset, WxDatasetClient


def cast(element):
    element = np.array(element, dtype=np.float32)
    tensor = th.FloatTensor(element)
    return tensor


class Model2d(nn.Module):
    def __init__(self, setting):
        super().__init__()
        self.setting = setting

        lsm = cast(load_lsm(setting.resolution, setting.root))
        slt = cast(load_slt(setting.resolution, setting.root))
        oro = cast(load_orography(setting.resolution, setting.root))
        aw = cast(load_area_weight(setting.resolution, setting.root))

        phi = cast(load_lat2d(setting.resolution, setting.root)) * np.pi / 180
        theta = cast(load_lon2d(setting.resolution, setting.root)) * np.pi / 180
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

        self.eva = Evaluator(setting.resolution, setting.root)

        self.dataset_train, self.dataset_test, self.dataset_eval = None, None, None
        self.train_size = -1
        self.test_size = -1
        self.eval_size = -1

        self.clipping_threshold = 3.0

    def load_dataset(self, phase, mode, **kwargs):
        if phase == 'train':
            if mode == 'server':
                self.dataset_train, self.dataset_eval = (
                        WxDataset(self.setting.root, self.setting.resolution,
                                  self.setting.years_train, self.setting.vars, self.setting.levels,
                                  input_span=self.setting.input_span, pred_shift=self.setting.pred_shift, pred_span=self.setting.pred_span, step=self.setting.step),
                        WxDataset(self.setting.root, self.setting.resolution,
                                  self.setting.years_eval, self.setting.vars, self.setting.levels,
                                  input_span=self.setting.input_span, pred_shift=self.setting.pred_shift, pred_span=self.setting.pred_span, step=self.setting.step)
                )
            else:
                ds_url = kwargs['url']
                self.dataset_train, self.dataset_eval = (
                    WxDatasetClient(ds_url, 'train', self.setting.resolution,
                              self.setting.years_train, self.setting.vars, self.setting.levels,
                              input_span=self.setting.input_span, pred_shift=self.setting.pred_shift,
                              pred_span=self.setting.pred_span, step=self.setting.step),
                    WxDatasetClient(ds_url, 'eval', self.setting.resolution,
                              self.setting.years_eval, self.setting.vars, self.setting.levels,
                              input_span=self.setting.input_span, pred_shift=self.setting.pred_shift,
                              pred_span=self.setting.pred_span, step=self.setting.step)
                )

            self.train_size = len(self.dataset_train)
            self.eval_size = len(self.dataset_eval)
        else:
            if mode == 'server':
                self.dataset_test = WxDataset(self.setting.root, self.setting.resolution,
                                              self.setting.years_test, self.setting.vars, self.setting.levels,
                                              input_span=self.setting.input_span, pred_shift=self.setting.pred_shift, pred_span=self.setting.pred_span, step=self.setting.step)
            else:
                ds_url = kwargs['url']
                self.dataset_test = WxDatasetClient(ds_url, 'test', self.setting.resolution,
                                              self.setting.years_test, self.setting.vars, self.setting.levels,
                                              input_span=self.setting.input_span, pred_shift=self.setting.pred_shift,
                                              pred_span=self.setting.pred_span, step=self.setting.step)

            self.test_size = len(self.dataset_test)

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
    def __init__(self, setting, enable_da=False):
        super().__init__(setting)
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
        raise NotImplementedError()
        return {}, None

    def get_targets(self, **kwargs):
        raise NotImplementedError()
        return {}, None

    def get_results(self, **kwargs):
        raise NotImplementedError()
        return {}, None

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
        return {}

    def lossfun(self, inputs, result, target):
        raise NotImplementedError()
        return 0.0


class Base3d(Base2d):
    def __init__(self, setting, enable_da=False):
        super().__init__(setting, enable_da)

    def augment_data(self, data):
        if self.enable_da and self.training:
            augmented = []
            b, c, t, w, h = data.size()
            for _ in range(b):
                slice = data[_:_+1]
                shift = self.lng_shift[_]
                flip = self.flip_status[_]
                slice = slice.roll(shift, dims=(4,))
                if flip == 1:
                    slice = th.flip(slice, dims=(3, 4))
                augmented.append(slice)
            data = th.cat(augmented, dim=0)
        return data

    def get_augmented_constant(self, input):
        b, c, t, w, h = input.size()
        constant = self.get_constant(input, input.device).repeat(b, 1, t, 1, 1)
        constant = self.augment_data(constant)
        phi = self.get_phi(input.device).repeat(b, 1, t, 1, 1)
        theta = self.get_theta(input.device).repeat(b, 1, t, 1, 1)
        constant = th.cat((constant, phi, theta), dim=1)
        return constant

    def get_inputs(self, **kwargs):
        raise NotImplementedError()
        return {}, None

    def get_targets(self, **kwargs):
        raise NotImplementedError()
        return {}, None

    def get_results(self, **kwargs):
        raise NotImplementedError()
        return {}, None

    def forward(self, *args, **kwargs):
        raise NotImplementedError()
        return {}

    def lossfun(self, inputs, result, target):
        raise NotImplementedError()
        return 0.0
