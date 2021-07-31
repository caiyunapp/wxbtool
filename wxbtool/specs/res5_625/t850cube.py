# -*- coding: utf-8 -*-

'''
 A modeling spec for t850

 This spec follows basic settings and discussions in

   Data-driven medium-range weather prediction with a Resnet pretrained on climate simulations:  A new model for WeatherBench
   by Stephan Rasp, Nils Thuerey
   https://arxiv.org/pdf/2008.08626.pdf

   but the spec was specialized for a 3d ResUNet

'''

import torch as th
import torch.nn as nn
from wxbtool.nn.model import Base3d
from wxbtool.specs.res5_625.t850rasp import SettingRasp
from wxbtool.data.variables import vars3d, code2var, split_name
from wxbtool.norms.meanstd import normalizors, denorm_t850


mse = nn.MSELoss()


class Setting3d4cube(SettingRasp):
    def __init__(self):
        super().__init__()
        self.step = 8                   # How many hours of a hourly step which all features in organized temporally
        self.input_span = 24             # How many hourly steps for an input
        self.pred_span = 1              # How many hourly steps for a prediction
        self.pred_shift = 72            # How many hours between the end of the input span and the beginning of prediction span


class Setting5d4cube(SettingRasp):
    def __init__(self):
        super().__init__()
        self.step = 8                   # How many hours of a hourly step which all features in organized temporally
        self.input_span = 24             # How many hourly steps for an input
        self.pred_span = 1              # How many hourly steps for a prediction
        self.pred_shift = 120           # How many hours between the end of the input span and the beginning of prediction span


class Spec(Base3d):
    def __init__(self, setting):
        super().__init__(setting)

        # following Rasp's schema from the above paper
        self.name = 't850_rasp4dim3'

    def get_inputs(self, **kwargs):
        vdic, vlst = {}, []
        for nm in self.setting.vars_in:
            c, l = split_name(nm)
            v = code2var[c]
            if v in vars3d:
                d = kwargs[v].view(-1, self.setting.input_span, self.setting.height, 32, 64)[:, :, self.setting.levels.index(l)]
            else:
                d = kwargs[v].view(-1, self.setting.input_span, 32, 64)
            d = d.view(-1, 1, self.setting.input_span, 32, 64)
            d = normalizors[nm](d)
            d = self.augment_data(d)
            vdic[nm] = d
            vlst.append(d)

        return vdic, th.cat(vlst, dim=1)

    def get_targets(self, **kwargs):
        t850 = kwargs['temperature'].view(-1, 1, self.setting.height, 32, 64)[:, :, self.setting.levels.index('850')]
        t850 = t850.view(-1, 1, 1, 32, 64)
        t850 = self.augment_data(t850)
        return {'t850': t850}, t850

    def get_results(self, **kwargs):
        t850 = denorm_t850(kwargs['t850'])
        return {'t850': t850}, t850

    def forward(self, **kwargs):
        raise NotImplementedError('Spec is abstract and can not be initialized')

    def lossfun(self, inputs, result, target):
        _, rst = self.get_results(**result)
        _, tgt = self.get_targets(**target)
        rst = self.weight * rst.view(-1, 1, 32, 64)
        tgt = self.weight * tgt.view(-1, 1, 32, 64)

        losst = mse(rst[:, 0], tgt[:, 0])

        return losst
