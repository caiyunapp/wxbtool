# -*- coding: utf-8 -*-

'''
 A modeling spec for t850

 This spec follows basic settings and discussions in

   Data-driven medium-range weather prediction with a Resnet pretrained on climate simulations:  A new model for WeatherBench
   by Stephan Rasp, Nils Thuerey
   https://arxiv.org/pdf/2008.08626.pdf

'''

import torch as th
import torch.nn as nn
from wxbtool.nn.model import Base2d
from wxbtool.nn.setting import Setting
from wxbtool.data.variables import vars3d, code2var, split_name
from wxbtool.norms.meanstd import normalizors, denorm_t850


mse = nn.MSELoss()


class SettingRasp(Setting):
    def __init__(self):
        super().__init__()
        self.resolution = '5.625deg'    # The spatial resolution of the model

        # Which vertical levels to choose
        self.levels = []
        # How many vertical levels to choose
        self.height = len(self.levels)

        # The name of variables to choose, for both input features and output
        self.vars = ['2m_temperature']

        # The code of variables in input features
        self.vars_in = ['t2m']

        # The code of variables in output
        self.vars_out = ['t2m']

        # temporal scopes for train
        self.years_train = [
            1980, 1981, 1982
        ]
        # temporal scopes for evaluation
        self.years_eval = [1980]
        # temporal scopes for test
        self.years_test = [1980]


class Setting3d(SettingRasp):
    def __init__(self):
        super().__init__()
        self.step = 8                   # How many hours of a hourly step which all features in organized temporally
        self.input_span = 3             # How many hourly steps for an input
        self.pred_span = 1              # How many hourly steps for a prediction
        self.pred_shift = 72            # How many hours between the end of the input span and the beginning of prediction span


class Setting5d(SettingRasp):
    def __init__(self):
        super().__init__()
        self.step = 8                   # How many hours of a hourly step which all features in organized temporally
        self.input_span = 3             # How many hourly steps for an input
        self.pred_span = 1              # How many hourly steps for a prediction
        self.pred_shift = 120           # How many hours between the end of the input span and the beginning of prediction span


class Spec(Base2d):
    def __init__(self, setting):
        super().__init__(setting)
        self.name = 't2m_test'

    def get_inputs(self, **kwargs):
        vdic, vlst = {}, []
        for nm in self.setting.vars_in:
            c, l = split_name(nm)
            v = code2var[c]
            if v in vars3d:
                d = kwargs[v].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index(l)]
            else:
                d = kwargs[v].view(-1, self.setting.input_span, 32, 64)
            d = normalizors[nm](d)
            d = self.augment_data(d)
            vdic[nm] = d
            vlst.append(d)

        return vdic, th.cat(vlst, dim=1)

    def get_targets(self, **kwargs):
        t2m = kwargs['2m_temperature'].view(-1, 1, 32, 64)
        t2m = self.augment_data(t2m)
        return {'t2m': t2m}, t2m

    def get_results(self, **kwargs):
        t2m = denorm_t2m(kwargs['t2m'])
        return {'t2m': t2m}, t2m

    def forward(self, **kwargs):
        raise NotImplementedError('Spec is abstract and can not be initialized')

    def lossfun(self, inputs, result, target):
        _, rst = self.get_results(**result)
        _, tgt = self.get_targets(**target)
        rst = self.weight * rst.view(-1, 1, 32, 64)
        tgt = self.weight * tgt.view(-1, 1, 32, 64)

        losst = mse(rst[:, 0], tgt[:, 0])

        return losst
